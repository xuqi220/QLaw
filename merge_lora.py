import os
import sys
sys.path.append("/home/kai/workspace")
import torch
import argparse
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelfiles.glm3.modeling_chatglm import ChatGLMForConditionalGeneration


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_model_dir', default="modelfiles/qwen2-7b", type=str, help='')
    parser.add_argument('--adapter_dir', default="llm_finetune/task/legal_concept_reasoning/ckp/epoch-3-step-450", type=str, help='')
    parser.add_argument('--output_dir', default="llm_finetune/task/legal_concept_reasoning/ckp/epoch-3-step-450", type=str, help='')
    return parser.parse_args()


if __name__ == '__main__':
    args = set_args()
    if os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        
    if "glm3" in args.ori_model_dir:
        base_model = ChatGLMForConditionalGeneration.from_pretrained(args.ori_model_dir, torch_dtype=torch.float16)
    if "qwen2" in args.ori_model_dir:
        base_model = AutoModelForCausalLM.from_pretrained(args.ori_model_dir,trust_remote_code=True, torch_dtype=torch.float16)

    lora_model = PeftModel.from_pretrained(base_model, args.adapter_dir, torch_dtype=torch.float16)
    lora_model.to("cpu")
    model = lora_model.merge_and_unload()
    model.save_pretrained( args.output_dir, max_shard_size="2GB")

