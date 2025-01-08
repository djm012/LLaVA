from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, CLIPImageProcessor
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import argparse

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token

def eval_model(args):
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    model = AutoAWQForCausalLM.from_quantized(
        args.model_path,
        fuse_layers=True,
        trust_remote_code=True,
        safetensors=True,
        use_cache=True,
        device_map="auto",
        use_flash_attention=False
    )
    
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    
    model.config.use_cache = True

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    # 统计变量
    total_input_tokens = 0
    total_output_tokens = 0
    total_time = 0

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        
        # 处理图像
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        
        if args.conv_mode == 'simple_legacy':
            prompt = qs
        else:
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        
        image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0)
        
        input_ids = input_ids.to(device)
        image_tensor = image_tensor.to(device, dtype=torch.float16)
        
        input_token_len = input_ids.shape[1]
        total_input_tokens += input_token_len
        
        # 去掉ASSISTANT:前面的部分，后面的才是text
        assistant_tokens = tokenizer.encode("ASSISTANT:", add_special_tokens=False)
        input_token_ids = input_ids[0].tolist()
        
        for i in range(len(input_token_ids) - len(assistant_tokens) + 1):
            if input_token_ids[i:i + len(assistant_tokens)] == assistant_tokens:
                start_idx = i + len(assistant_tokens)
                break
        
        # 修改 prepare_inputs_for_generation 来跳过之前的 tokens
        with torch.inference_mode():
            def prepare_inputs(*args, **kwargs):
                # 只保留 ASSISTANT: 之后的部分
                current_input_ids = args[0]
                if current_input_ids.shape[1] <= start_idx:
                    current_attention_mask = kwargs.get("attention_mask", None)
                    if current_attention_mask is not None:
                        current_attention_mask = current_attention_mask[:, :start_idx]
                    return {
                        "input_ids": current_input_ids,
                        "attention_mask": current_attention_mask,
                        "position_ids": kwargs.get("position_ids", None),
                        "images": image_tensor,
                        "use_cache": True,
                        "past_key_values": kwargs.get("past_key_values", None)
                    }
                return None
            
            model.prepare_inputs_for_generation = prepare_inputs
            
            # 开始计时
            start_time = time.time()
            
            # 生成输出
            output_ids = model.generate(
                input_ids[:, :start_idx],  # 只使用到 ASSISTANT: 的部分
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        generation_time = time.time() - start_time
        
        # 直接解码生成的部分
        outputs = tokenizer.decode(output_ids[0, start_idx:], skip_special_tokens=True).strip()
        
        # 统计
        output_token_len = len(tokenizer.encode(outputs))

        total_output_tokens += output_token_len
        
        total_time += generation_time
        print("total_time",total_time)

        # 记录统计信息
        stats = {
            "input_tokens": input_token_len,
            "output_tokens": output_token_len,
            "process_time": generation_time,
            "tokens_per_second": (output_token_len+input_token_len) / generation_time
        }
        
        print(f"\nImage {idx} Statistics:")
        print(f"Input tokens: {input_token_len}")
        print(f"Generated tokens: {output_token_len}")
        print(f"Processing time: {generation_time:.2f} seconds")
        print(f"Generation speed: {stats['tokens_per_second']:.2f} tokens/sec")
        print("-" * 50)
        
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": shortuuid.uuid(),
            "model_id": args.model_path,
            "metadata": stats
        }) + "\n")
    
    print("\nOverall Processing Statistics:")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total generated tokens: {total_output_tokens}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average generation speed: {(total_output_tokens + total_input_tokens) / total_time:.2f} tokens/sec")
    
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, default="answers.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    
    eval_model(args)

