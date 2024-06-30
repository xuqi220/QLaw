import argparse
import torch
import json
from glm3.modeling_chatglm import ChatGLMForConditionalGeneration
from glm3.tokenization_chatglm import ChatGLMTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--device", type=str, default="0", help="")
    parser.add_argument("--mode", type=str, default="glm3", help="")
    parser.add_argument("--model_path", type=str, default="modelfile_path", help="")
    parser.add_argument("--max_length", type=int, default=2048, help="")
    parser.add_argument("--do_sample", type=bool, default=True, help="")
    parser.add_argument("--top_p", type=float, default=0.8, help="")
    parser.add_argument("--temperature", type=float, default=0.8, help="")
    return parser.parse_args()

def load_test_sample(path="test_data_path", idx=0):
    data = []
    with open(path, "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            data.append(json.loads(line))
    return data[0]
        

def predict_one_sample(instruction, input, model, tokenizer, args):
    result, _ = model.chat(tokenizer, 
                           instruction + input,
                           max_length=args.max_length, 
                           do_sample=args.do_sample,
                           top_p=args.top_p, 
                           temperature=args.temperature)
    return result


if __name__ == '__main__':
    args = parse_args()
    model = ChatGLMForConditionalGeneration.from_pretrained(args.model_path, 
                                                            device_map="auto",
                                                            torch_dtype=torch.float16
                                                            )
    tokenizer = ChatGLMTokenizer.from_pretrained(args.model_path)
    sample = load_test_sample()
    instruction = sample['instruction']
    input = sample['input']
    print(f"问题：{instruction}\n")
    print(f"事实描述： {input}\n")
    r = predict_one_sample(instruction, input, model, tokenizer, args)
    print(f"答案：{r}")
    print(f"参考：{sample['output']}")