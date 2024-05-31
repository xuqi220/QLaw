
import deepspeed
import argparse
import torch
import json
import math
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from utils import set_random_seed, print_rank_0, GLMPromptDataSet, DataCollator, print_trainable_parameters, to_device, save_model
from glm3.modeling_chatglm import ChatGLMForConditionalGeneration
from glm3.tokenization_chatglm import ChatGLMTokenizer 
from glm3.configuration_chatglm import ChatGLMConfig


def arg_parse():
    parser = argparse.ArgumentParser()
    # deepspeed config
    parser.add_argument("--ds_file", type=str, default="/lora_sft/ds_zero2_no_offload.json")
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    # dataset
    parser.add_argument("--train_file", type=str, default="") # 
    parser.add_argument("--max_len", type=int, default=1560)
    parser.add_argument("--max_src_len", type=int, default=1024)
    parser.add_argument("--is_skip", action='store_true')
    # model
    parser.add_argument("--model_path", type=str, default="/glm3") # 模型参数文件
     # lora
    parser.add_argument("--lora_dim", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_module_name", type=str, default="query_key_value,dense_h_to_4h,dense_4h_to_h,dense")
    # train
    parser.add_argument("--output_dir", type=str, default="solver") 
    parser.add_argument("--train_batch_size_per_device", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_train_epoch", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2333, help="")
    parser.add_argument("--show_loss_step", default=10, type=int, help="")
    parser.add_argument("--save_model_step", default=None, type=int, help="")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def main():
    # 解析参数
    args = arg_parse()
    # 设置当前进程可见设备
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda:{args.local_rank}")
    # 初始化分布式环境
    deepspeed.init_distributed()
    args.global_rank = torch.distributed.get_rank() # 0~word_size
    
    # print(f"local rank: {args.local_rank}  global rank: {args.global_rank}")
        
    set_random_seed(args.seed)
    # print(f"{args.local_rank} arrive barrier")
    # 阻塞进程，直到所有的进程到达
    torch.distributed.barrier() 
    
    # 加载tokenizer
    tokenizer = ChatGLMTokenizer.from_pretrained(args.model_path)
    print_rank_0(f"tokenizer.pad_token: {tokenizer.pad_token}")
    print_rank_0(f"tokenizer.eos_token: {tokenizer.eos_token}")
    
    # 加载lora模型
    model = ChatGLMForConditionalGeneration.from_pretrained(args.model_path)
    config = LoraConfig(
        r=args.lora_dim,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_module_name.split(','),
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )
    model = get_peft_model(model, config)
    model.config.torch_dtype = torch.float32
    print_trainable_parameters(model)
    # 加载数据集
    train_dataset = GLMPromptDataSet(
        data_path=args.train_file,
        tokenizer=ChatGLMTokenizer.from_pretrained(args.model_path),
        max_len=1560,
        max_src_len=1024,
        is_skip=False
    )
    train_sampler = DistributedSampler(train_dataset) # 分布式采样
    data_collator = DataCollator(tokenizer)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=data_collator,
        sampler=train_sampler,
        batch_size=args.train_batch_size_per_device
    )
    
    # 加载deepspeed配置文件,从commandline获取deepspeed环境参数
    with open(args.ds_file, "r", encoding="utf-8") as fi:
        ds_config = json.load(fi)
    
    ds_config['train_micro_batch_size_per_gpu'] = args.train_batch_size_per_device
    ds_config['train_batch_size'] = args.train_batch_size_per_device * torch.distributed.get_world_size() * args.gradient_accumulation_steps
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    ds_config['optimizer']['params']['lr'] = args.learning_rate
    ds_config['optimizer']['params']['betas'] = (0.9, 0.95)
    ds_config['optimizer']['params']['eps'] = 1e-8
    ds_config['optimizer']['params']['weight_decay'] = 0.1
    num_train_steps = args.num_train_epoch * math.ceil(len(train_dataloader)/args.gradient_accumulation_steps)
    num_warmup_steps = int(args.warmup_ratio*num_train_steps)
    ds_config['scheduler']['params']['total_num_steps'] = num_train_steps
    ds_config['scheduler']['params']['warmup_num_steps'] = num_warmup_steps
    ds_config['scheduler']['params']['warmup_max_lr'] = args.learning_rate
    ds_config['scheduler']['params']['warmup_min_lr'] = 0.1*args.learning_rate
    
    # 初始化deepspeed
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        args=args,
        config=ds_config,
        dist_init_required=True
    )
    # print(f"type of model {type(model)}")
    global_step = 0
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    for epoch in range(args.num_train_epoch):
        print_rank_0(f"Begining of Epoch {epoch+1}/{args.num_train_epoch}, Total Micro Batches {len(train_dataloader)}", args.local_rank)
        model.train()
        process_bar = tqdm(range(len(train_dataloader)))
        for mini_step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            tr_loss += loss.item()
            model.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model.step()
            process_bar.update(1)
            if model.is_gradient_accumulation_boundary() and args.local_rank <= 0:
                global_step+=1
                # 记录loss
                if global_step % args.show_loss_step == 0:
                    print_rank_0(f"Epoch: {epoch} | "  
                                f"mini_step: {mini_step + 1} | "
                                f"global_step:{global_step} | " 
                                f"loss: {(tr_loss - logging_loss)/(args.show_loss_step * args.gradient_accumulation_steps)} | "
                                ,args.local_rank)
                    print_rank_0(f"step: {mini_step + 1}-{global_step}-{model.global_steps}", args.global_rank)
                    if args.global_rank <= 0:
                        logging_loss = tr_loss
        if args.global_rank <= 0:
            save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}-step-{global_step}")
                
if __name__ == "__main__":
    main()