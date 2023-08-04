#!/usr/bin/env bash
#/usr/lib/software/spark/spark-2.1.2-bin-U1/bin/task_processor --master yarn-client --queue root.spark.hdp_ubu_spider.spark --py-files test_spark.py

#OK local python
#spark-submit --master yarn --deploy-mode cluster --queue root.spark.hdp_ubu_spider.spark --executor-cores 2 --executor-memory 4G --num-executors 1 --py-files test_spark.zip test_spark.py
#TASKNAME=youliao_bert
TASKNAME=tribe_labels
TASKID=20190710

# shellcheck disable=SC2164
cd ../../src/text_classification
echo "enter AISEALs/src/text_classification"

DEPENDENCY="dependency.zip"
if [ ! -f $DEPENDENCY ];then
    echo "$DEPENDENCY not exists"
    #zip -r -9 -q $DEPENDENCY ./* --exclude files/*
    echo "start package dependency files"
    zip -r -9 -q $DEPENDENCY ./*
    echo "finnish package dependency files"
else
    echo "$DEPENDENCY exists"
fi

#spark-submit --py-files $DEPENDENCY task_processor/task_processor_spark.py "/Users/jiananliu/AISEALs/data/text_classification" $TASKNAME $TASKID

#conda python3.5
#spark-submit \
/usr/lib/software/spark/spark-2.3.2-bin-U1/bin/spark-submit \
	--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python \
	--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=./environment/bin/python \
	--conf spark.dynamicAllocation.maxExecutors=16 \
	--master yarn \
	--deploy-mode cluster \
	--queue root.spark.hdp_ubu_spider.spark \
	--executor-cores 4 --executor-memory 16G --num-executors 8 \
	--archives /home/hdp_ubu_spider/spark/dl/pyspark/py3spark.tar.gz#environment \
	--files files/stop_words_ch_utf8.txt,files/vocab.txt \
	--py-files $DEPENDENCY task_processor/task_processor_spark.py --base_path="/home/hdp_ubu_spider/resultdata/work/text_classification/" --task_name=$TASKNAME --task_id=$TASKID --debug_mode=False

#conda shell
#task_processor \
#	--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python \
#	--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=./environment/bin/python \
#	--master yarn \
#	--deploy-mode client \
#	--queue root.spark.hdp_ubu_spider.spark \
#	--executor-cores 2 --executor-memory 4G --num-executors 1 \
#	--archives py3spark.tar.gz#environment \
#	--py-files tokenization.py,youliao_bert_processor.py,data_processor.py
