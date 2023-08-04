#export MODEL_BASE_DIR=/opt/crawler/opt/zhoujianbin/ljn/AISEALs/models
#export DATA_DIR=/opt/crawler/opt/zhoujianbin/ljn/AISEALs/data
#online
export MODEL_BASE_DIR=../model/cnn_text_classification/output
export DATA_DIR=../data
python run_classifier.py \
	--task_name=youliao \
	--do_train=false \
	--do_eval=true \
	--etl_rawdata=false \
	--etl_tfrecord=false \
	--data_dir=$DATA_DIR/cnn_text_classification/youliao_raw_data/ \
	--vocab_file=$MODEL_BASE_DIR/vocab.txt \
	--bert_config_file=$MODEL_BASE_DIR/bert_config.json \
	--init_checkpoint=$MODEL_BASE_DIR/model.ckpt-31876 \
	--max_seq_length=128 \
	--train_batch_size=32 \
	--eval_batch_size=64 \
	--learning_rate=5e-5 \
	--num_train_epochs=2.0 \
	--output_dir=./runs/
