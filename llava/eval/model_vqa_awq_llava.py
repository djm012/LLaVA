import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import time

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from awq import AutoAWQForCausalLM
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print("=========================================DEFAULT_IMAGE_TOKEN=====================================================",DEFAULT_IMAGE_TOKEN)
        # print("=========================================prompt=====================================================",prompt)

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,device_map='auto',load_4bit=args.is_4bit,load_8bit=args.is_8bit)
    # print('image_processor',image_processor)
    """
    model_quantized  = autoawq().. llmforcallm
    model.layers = modelquantized.blocks     (etc.)属性替换
    或者 直接set属性
    """
    model_quantized = AutoAWQForCausalLM.from_quantized('./checkpoints/awq_model', fuse_layers=True,
                                          trust_remote_code=False, safetensors=True)
    model.model.layers = model_quantized.model.model.blocks


    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # print('--------------------------------------------model----------------------------------------------',model)
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    total_input_tokens = 0
    total_output_tokens = 0
    total_time = 0

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        print('=============================================================================================')

        # 计算输入token数量
        input_token_len = input_ids.shape[1]
        total_input_tokens += input_token_len

        # 记录开始时间
        start_time = time.time()
        
        with torch.inference_mode():
            print('=========================================input_ids.shape=====================================================',input_ids.shape)
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
        
        # 计算处理时间
        end_time = time.time()
        process_time = end_time - start_time
        total_time += process_time
        
        # 计算输出token数量
        output_token_len = output_ids.shape[1]
        total_output_tokens += output_token_len
        print('=========================================output_ids.shape=====================================================',output_ids)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # 记录每张图的统计信息
        stats = {
            "input_tokens": input_token_len,
            "output_tokens": output_token_len,
            "process_time": process_time,
            "tokens_per_second": (input_token_len + output_token_len) / process_time
        }

        # 打印每张图的统计信息
        print(f"\nImage {idx} Statistics:")
        print(f"Input tokens: {input_token_len}")
        print(f"Output tokens: {output_token_len}")
        print(f"Processing time: {process_time:.2f} seconds")
        print(f"Tokens per second: {(input_token_len + output_token_len) / process_time:.2f}")
        print("-" * 50)
        print('=============================================================================================')

        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": shortuuid.uuid(),
            "model_id": model_name,
            "metadata": stats
        }) + "\n")

    # 打印总体统计信息
    print("\nProcessing Statistics:")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average tokens per second: {(total_input_tokens + total_output_tokens) / total_time:.2f}")

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--is_4bit", type=bool, default=False)
    parser.add_argument("--is_8bit", type=bool, default=False)
    args = parser.parse_args()

    eval_model(args)
