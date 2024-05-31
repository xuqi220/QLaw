import os
import json
import torch
import random
import numpy as np
from transformers import set_seed
from torch.utils.data import Dataset, DataLoader
from glm3.tokenization_chatglm import ChatGLMTokenizer


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)
        
def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output
        
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print("trainable params: {} || all params: {} || trainable%: {}".format(trainable_params, all_param,
                                                                            100 * trainable_params / all_param))

def save_model(model, tokenizer, output_dir, model_name, state_dict=None):
    save_dir = os.path.join(output_dir, model_name)
    if state_dict == None:
        model.save_pretrained(save_dir, torch_dtype=torch.float16)
    else:
        model.save_pretrained(save_dir, state_dict=state_dict, torch_dtype=torch.float16)
    tokenizer.save_pretrained(save_dir)
    

class GLMPromptDataSet(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len, is_skip):
        self.all_data = []
        skip_data_number = 0
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                skip_flag = False
                t1 = [tokenizer.get_command("<|user|>")]
                t2 = tokenizer.encode("\n", add_special_tokens=False)
                t3 = tokenizer.encode(sample["instruction"] + sample["input"], add_special_tokens=False)
                src_tokens =  t1 + t2 + t3
                             
                if len(src_tokens) > max_src_len:
                    # 当输入内容超长时，随向后截断
                    src_tokens = src_tokens[:max_src_len]
                    skip_flag = True

                max_tgt_len = max_len - 6 - len(src_tokens)
                tgt_tokens = [tokenizer.get_command("<|assistant|>")] + tokenizer.encode("\n", add_special_tokens=False) + \
                             tokenizer.encode(sample["output"], add_special_tokens=False)

                if len(tgt_tokens) > max_tgt_len:
                    # 当输出内容超长时，随向后截断
                    tgt_tokens = tgt_tokens[:max_tgt_len]
                    skip_flag = True

                # ChatGLM3需要增加[gMASK]、sop两个标记
                t4 = [tokenizer.get_command("[gMASK]"), tokenizer.get_command("sop")]
                input_ids =  t4 + src_tokens + tgt_tokens + [tokenizer.eos_token_id]
                
                context_length = len(src_tokens) + 2
                labels = [-100] * context_length + input_ids[context_length:]

                assert len(input_ids) == len(labels)
                assert len(input_ids) <= max_len
                if is_skip and skip_flag:
                    skip_data_number += 1
                    continue
                self.all_data.append({"input_ids": input_ids, "labels": labels})
        print("the number of skipping data is {}".format(skip_data_number))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance
    

class DataCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        lengths = [len(instance["input_ids"]) for instance in batch]
        batch_max_len = max(lengths)

        input_ids_batch, labels_batch = [], []
        for instance in batch:
            input_ids = instance["input_ids"]
            labels = instance["labels"]

            padding_len = batch_max_len - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_len
            labels = labels + [-100] * padding_len

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)

        return {"input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
                "labels": torch.tensor(labels_batch, dtype=torch.long)}

if __name__=="__main__":
    GLMPromptDataSet(
        data_path="solver_sft.json",
        tokenizer=ChatGLMTokenizer.from_pretrained("glm3"),
        max_len=1560,
        max_src_len=1024,
        is_skip=False
    )
    