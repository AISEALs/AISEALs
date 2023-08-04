#!/usr/bin/env bash
export DATA_DIR=/Users/jiananliu/AISEALs/data/cnn_text_classification
#export DATA_DIR=/home/work/ljn/
#export DATA_DIR=/tmp/AISEALs/text_classification/
export INIT_CHECKPOINT=/Users/jiananliu/AISEALs/model/cnn_text_classification/runs/model.ckpt-1074

python -u text_cnn_model/youliao_estimators.py \
	--task_id=0 \
	--data_dir=$DATA_DIR \
	--init_checkpoint=$INIT_CHECKPOINT \
	--export_model=True \
	--train_batch_size=64  \
	--eval_batch_size=64  \
	--num_checkpoints=2  \
	--max_steps=200000 \
	--l2_reg_lambda=0.1 \
	--output_dir=runs
	#--output_dir=/Users/jiananliu/AISEALs/models/text_classification/runs/
