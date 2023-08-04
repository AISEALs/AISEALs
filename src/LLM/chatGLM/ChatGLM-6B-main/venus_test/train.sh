PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=6 python3 main.py \
    --do_train \
    --train_file /group/30106/summerwuxia/trlx-main/data/datasets/AdvertiseGen/train.json \
    --validation_file /group/30106/summerwuxia/trlx-main/data/datasets/AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /group/30147/summerwuxia/model/chatglm-6b/models--THUDM--chatglm-6b/snapshots/aa51e62ddc9c9f334858b0af44cf59b05c70148a \
    --output_dir /group/30147/summerwuxia/model/checkpoint/adgen-chatglm-6b/adgen-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

