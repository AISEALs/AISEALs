
LR=1e-4

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py \
    --do_train \
    --train_file /group/30106/summerwuxia/trlx-main/data/datasets/AdvertiseGen/train.json \
    --test_file /group/30106/summerwuxia/trlx-main/data/datasets/AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /group/30147/summerwuxia/model/chatglm-6b/models--THUDM--chatglm-6b/snapshots/aa51e62ddc9c9f334858b0af44cf59b05c70148a \
    --output_dir /group/30147/summerwuxia/model/checkpoint/adgen-chatglm-6b/adgen-chatglm-6b-ft-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --pre_seq_len 64 \
    --predict_with_generate \
    --max_steps 100000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --fp16 False

