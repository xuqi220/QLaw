CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port 29500 llm_finetune/train.py \
                --ds_file llm_finetune/ds_zero2_no_offload.json \
                --train_file llm_finetune/task/legal_concept_reasoning/solver_sft_small.json \
                --max_len 1560 \
                --max_src_len 1500 \
                --model_path modelfiles/glm3 \
                --lora_dim 16 \
                --lora_alpha 64 \
                --lora_dropout 0.1 \
                --lora_module_name "query_key_value,dense_h_to_4h,dense_4h_to_h" \
                --output_dir llm_finetune/task/legal_concept_reasoning/ckp \
                --train_batch_size_per_device 4 \
                --gradient_accumulation_steps 4 \
                --learning_rate 1e-5 \
                --weight_decay 0.1 \
                --num_train_epoch 5 \
                --warmup_ratio 0.1 \
                --seed 2333 \
                --show_loss_step 100 \
                --save_model_step 100