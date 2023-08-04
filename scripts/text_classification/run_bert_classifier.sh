#!/usr/bin/env bash

#深度学习平台中,运行脚本
#hadoop fs -get /home/hdp_ubu_spider/resultdata/work/text_classification/scripts.tar.bz2 && tar -xf scripts.tar.bz2 && ./AISEALs/scripts/text_classification/run_bert_classifier.sh tribe_labels 20190708

if [ $# -lt 2 ];then
    echo "please run script with params.\neg: run_bert_classifier \${task_name} \${task_id}"
    exit 1
else
    TASK_NAME=$1
    TASK_ID=$2
    echo "task_name: ${TASK_NAME} task_id: ${TASK_ID}"
fi

TASK_TOTAL_NAME=${TASK_NAME}_${TASK_ID}
TASK_TOTAL_PATH=AISEALs/data/text_classification/$TASK_TOTAL_NAME

cd /tmp/;
echo "enter /tmp"

hadoop fs -get /home/hdp_ubu_spider/resultdata/work/text_classification/AISEALs.tar.bz2 && tar -xf AISEALs.tar.bz2 && rm -f AISEALs.tar.bz2;
echo "get AISEALs.tar.bz2 success"

mkdir -p $TASK_TOTAL_PATH && hadoop fs -get /home/hdp_ubu_spider/resultdata/work/text_classification/$TASK_TOTAL_NAME/tfrecord ./$TASK_TOTAL_PATH && hadoop fs -get /home/hdp_ubu_spider/resultdata/work/text_classification/$TASK_TOTAL_NAME/label_classes.csv ./$TASK_TOTAL_PATH
echo "get tfrecord and label_classes.csv into $TASK_TOTAL_PATH"

#online
#export MODEL_BASE_DIR=/tmp/AISEALs/models/text_classification/bert/chinese_L-12_H-768_A-12
export BASE_DIR=/tmp/AISEALs/src/text_classification/
export DATA_DIR=/tmp/AISEALs/data/text_classification/
#INIT_CHECKPOINT=$MODEL_BASE_DIR/bert_model.ckpt
#使用上线预置模型功能
export MODEL_BASE_DIR=/workspace/train/preTrainingModel/chinese_L-12_H-768_A-12
INIT_CHECKPOINT=$MODEL_BASE_DIR/bert_model.ckpt

#debug
#export USER_DIR=/Users/jiananliu
#export USER_DIR=/tmp
#export MODEL_BASE_DIR=$USER_DIR/AISEALs/models/text_classification/bert/chinese_L-12_H-768_A-12
#export BASE_DIR=$USER_DIR/AISEALs/text_classification
#export DATA_DIR=$USER_DIR/AISEALs/data/text_classification
##INIT_CHECKPOINT=/Users/jiananliu/AISEALs/models/text_classification/runs/models.ckpt-31008
#INIT_CHECKPOINT=$MODEL_BASE_DIR/bert_model.ckpt
cd $BASE_DIR
echo "enter $BASE_DIR"

echo "starting run bert/run_classifier.py"

python -u models/bert/run_classifier.py \
	--task_name=$TASK_NAME \
	--task_id=$TASK_ID \
	--do_train=true \
	--do_eval=true \
	--export_model=true \
	--base_dir=$DATA_DIR \
	--vocab_file=$BASE_DIR/files/vocab.txt \
	--bert_config_file=$MODEL_BASE_DIR/bert_config.json \
	--init_checkpoint=$INIT_CHECKPOINT \
	--max_seq_length=128 \
	--train_batch_size=32 \
	--eval_batch_size=32 \
	--learning_rate=5e-5 \
	--num_train_epochs=1.0 \
	--output_dir=$BASE_DIR/runs/
	#--output_dir=/tmp/AISEALs/text_classification/runs/

#python bert/run_classifier.py --task_name=youliao --do_train=true --do_eval=false  --data_dir=../data/text_classification/ --vocab_file=../models/text_classification/bert/chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=../models/text_classification/bert/chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=../models/text_classification/bert/chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=5e-5 --num_train_epochs=2.0 --output_dir=./output/
