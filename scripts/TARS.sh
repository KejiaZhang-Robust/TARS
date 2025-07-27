#!/bin/bash
unset CUDA_VISIBLE_DEVICES

export CUDA_LAUNCH_BLOCKING=1 

deepspeed --master_addr=127.0.0.1 --master_port=29501 --include=localhost:0,1,2,3,4,5,6,7 llava/train/train_mem.py \
   --deepspeed ./scripts/zero3.json \
   --model_name_or_path  llava_path \
   --version v1 \
   --data_path RLHF-V-Dataset.parquet \
   --image_folder / \
   --vision_tower vision_tower_path \
   --mm_projector_type mlp2x_gelu \
   --mm_vision_select_layer -2 \
   --mm_use_im_start_end False \
   --mm_use_im_patch_token False \
   --image_aspect_ratio pad \
   --group_by_modality_length True \
   --bf16 True \
   --output_dir output_path \
   --num_train_epochs 3 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 8 \
   --evaluation_strategy "no" \
   --save_strategy "epoch" \
   --save_steps 1000 \
   --save_total_limit 10 \
   --learning_rate 5e-7 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --tf32 True \
   --model_max_length 2048 \
   --gradient_checkpointing True \
   --max_grad_norm 20.0 \
   --dataloader_num_workers 4 \
   --lazy_preprocess True \
   --report_to none \
   --task DPO \
   --use_image_type diffusion \
   --diffusion_step 500 \
   --dpo_token_weighted False \
   --dpo_token_weight 4.0 \
   --use_cross_modal_loss False \
   --use_tdpo False \
   --tok_beta 0.1