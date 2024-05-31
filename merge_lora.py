import torch
import argparse
from peft import PeftModel
from glm3.modeling_chatglm import ChatGLMForConditionalGeneration


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_model_dir', default="glm3", type=str, help='')
    parser.add_argument('--model_dir', default="solver/epoch-1-step-50", type=str, help='')
    return parser.parse_args()


if __name__ == '__main__':
    args = set_args()
    base_model = ChatGLMForConditionalGeneration.from_pretrained(args.ori_model_dir, torch_dtype=torch.float16)
    lora_model = PeftModel.from_pretrained(base_model, args.model_dir, torch_dtype=torch.float16)
    lora_model.to("cpu")
    model = lora_model.merge_and_unload()
    ChatGLMForConditionalGeneration.save_pretrained(model, args.model_dir, max_shard_size="2GB")
