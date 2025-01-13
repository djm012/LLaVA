python -m llava.eval.model_vqa_awq_llava \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/test_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b-awq.jsonl \
    --temperature 0.7 \
    --conv-mode vicuna_v1
    # --projector-path ./checkpoints/llava-v1.5-13b-pretrain/ \

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b-awq.jsonl
