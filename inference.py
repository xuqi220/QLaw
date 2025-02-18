import argparse
import torch
import json
import random
import sys
sys.path.append("/home/kai/workspace")
from modelfiles.glm3.modeling_chatglm import ChatGLMForConditionalGeneration
from modelfiles.glm3.tokenization_chatglm import ChatGLMTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, default="llm_finetune/task/legal_concept_reasoning/solver_sft_small.json", help="")
    # Model
    parser.add_argument("--device", type=str, default="0", help="")
    parser.add_argument("--model_name", type=str, default="qwen2", help="")
    parser.add_argument("--ckp", type=str, default="llm_finetune/task/legal_concept_reasoning/ckp/epoch-3-step-450", help="")
    parser.add_argument("--max_length", type=int, default=2048, help="")
    parser.add_argument("--do_sample", type=bool, default=True, help="")
    parser.add_argument("--top_p", type=float, default=0.8, help="")
    parser.add_argument("--temperature", type=float, default=0.8, help="")
    return parser.parse_args()

def load_test_samples(path, k=1):
    with open(path, "r", encoding="utf-8") as fi:
        data = [json.loads(s) for s in random.sample(fi.readlines(), k=k)]
    return data
        

def generate(instruction, input, model, tokenizer, args):
     if "glm3" == args.model_name:
        result, _ = model.chat(tokenizer, 
                            instruction + input,
                            max_length=args.max_length, 
                            do_sample=args.do_sample,
                            top_p=args.top_p, 
                            temperature=args.temperature)
     if "qwen2" == args.model_name:
        ipt = tokenizer(f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n" + "<|im_start|>Assistant: ", return_tensors="pt").to(model.device)
        result = tokenizer.decode(model.generate(**ipt, max_length=2048, do_sample=True, eos_token_id=tokenizer.eos_token_id, temperature=0.1)[0], skip_special_tokens=True)
     return result


if __name__ == '__main__':
    args = parse_args()
    if "glm3" == args.model_name:
        model = ChatGLMForConditionalGeneration.from_pretrained(args.ckp, torch_dtype=torch.float16,device_map=f"cuda:{args.device}")
        tokenizer = ChatGLMTokenizer.from_pretrained(args.ckp)
    if "qwen2" == args.model_name:
        model = AutoModelForCausalLM.from_pretrained(args.ckp, torch_dtype=torch.float16,device_map=f"cuda:{args.device}")
        tokenizer = AutoTokenizer.from_pretrained(args.ckp)
    model.eval()
    samples = load_test_samples(args.test_data_path)
    for sample in samples:
        instruction = sample['instruction']
        input = sample['input']
        print(f"instruction：{instruction}\n")
        print(f"User： {input}\n")
        r = generate(instruction, input, model, tokenizer, args)
        print(f"response：{r}")