<img src="./assets/qlaw.png" style="height:150px">

# QLaw-中文法律大模型
QLaw项目旨在利用大模型为大众提供更加专业的中文法律服务，和大家交流大模型的相关的知识，从而为社区提供更加优质的模型。欢迎大家follow，期待和大家一起学习进步。

⚡ 为复杂的多被告人指控预测任务设计的Benchmark已被ACL2024主会录用（即将开源）。

## 免责声明

1. 本项目资源**仅供学术研究使用，严禁任何商业用途**。
2. 当前本项目当前无法保证其推理结果准确性，**严禁用于真实法律场景**。
3. 本项目不承担任何法律责任，以及因使用本项目相关资源而产生的任何损失。

## 项目介绍
本项目以GLM3-6B作为基座模型，利用DeepSpeed分布式训练工具以及Lora、P-tuning等技术，在法律领域数据集上进行微调。本项目分期进行，不断更新，欢迎一起学习进步！！！

### 第一期
理解法律概念是法律推理（Legal Reasoning），法律知识现实场景应用的基石。我们从[中国裁判文书网](https://wenshu.court.gov.cn/) 收集了大量的刑事案件文书(已脱敏)，并抽取出适用的101条常用的法律概念，例如：自首、累犯、盗窃罪等等，以及针对现实场景的相应解释。数据格式如下：
```
{
"instruction": "你是一个律师，请解释为什么被告人王某平构成累犯?", 

"input": "王某平，1979年4月19日出生，贵州省xxx人，文盲，农民，住贵州省xxx。因犯盗窃罪，2003年2月26日被浙江省长兴县人民法院判处有期徒刑八年六个月；...2019年2月19日因犯盗窃罪被浙江省德清县人民法院判处有期徒刑一年四个月，并处罚金人民币一万二千元，于2019年8月17日刑满释放。经审理查明，2019年10月28日凌晨，王某平从海宁市许村镇翁埠村罗化庵出发先后窜至许村镇前进村前进小区31号、前进村瑞丰庵25号，采用搭线等手段窃得被害人梁某、龙某的电动车各1辆，...足以认定。",

"output": "王某平在有期徒刑执行完毕后五年内再犯应当判处有期徒刑以上刑罚之罪，是累犯，依法应当从重处罚"
}
```
我们在该数据集上进行了微调，可利用本项目提供的`merge_lora.py`文件合并lora权重。另外我们也提供了完整权重的[百度网盘链接](https://pan.baidu.com/s/1gjObw73ZUWIN0KOGSLjlYw?pwd=hlyv)。

* Lora 训练方式：
  ```
  CUDA_VISIBLE_DEVICES=0,1,2 deepspeed --master_port 29500 train.py \
                --ds_file ds_zero2_no_offload.json \
                --train_file data/solver_sft.json \
                --max_len 1560 \
                --max_src_len 1024 \
                --model_path glm3/ \
                --lora_dim 16 \
                --lora_alpha 64 \
                --lora_dropout 0.1 \
                --lora_module_name "query_key_value,dense_h_to_4h,dense_4h_to_h,dense" \
                --output_dir ./solver \
                --train_batch_size_per_device 1 \
                --gradient_accumulation_steps 4 \
                --learning_rate 1e-5 \
                --weight_decay 0.1 \
                --num_train_epoch 3 \
                --warmup_ratio 0.1 \
                --seed 2333 \
                --show_loss_step 50 \
                --save_model_step 50
  ```













## 致谢
LexiLaw https://github.com/CSHaitao/LexiLaw

LawGPT https://github.com/LiuHC0428/LAW-GPT
