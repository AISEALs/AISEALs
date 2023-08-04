#/user/bin

#cd cpp & g++ es_noise.cpp -std=gnu++11

# sys env
#BASE_PATH=mdfs://cloudhdfs/mttsparknew/data/video/yanghaozhou/miniFloat/Es
BASE_PATH=mdfs://cloudhdfs/pcfeeds/qbdata/nemoztwang/MiniVideo/Float/Praise
HADOOP=/data/home/hadoop-venus/bin/hadoop

MDFS_PATH=$BASE_PATH/${MODEL_VERSION}
FEATURE_LIST_PATH=$BASE_PATH/${MODEL_VERSION}_featureList
MODEL_BAK_PATH=$BASE_PATH/${MODEL_VERSION}_modelbak

DCACHE_KEY=${MODEL_VERSION:2}
echo "------------$(date +"%Y-%m-%d %H:%M:%S")----------------"
echo "MODEL_VERSION: $MODEL_VERSION"
echo "START_DATA_HOUR: $START_DATA_HOUR"
echo "DACHE_KEY: $DCACHE_KEY"
echo "PART:" $PART
echo "LR: $LR"
echo "SIGMA: $SIGMA"
echo "MDFS_PATH: $MDFS_PATH"
echo "FEATURE_LIST_PATH: $FEATURE_LIST_PATH"
echo "MODEL_BAK_PATH: $MODEL_BAK_PATH"

START_DATA_HOUR_F1="${START_DATA_HOUR:0:10} ${START_DATA_HOUR:11}" # yyyy-mm-dd HH
START_DATA_HOUR_F2=$(date -d "$START_DATA_HOUR_F1" +"%Y-%m-%d/%H")  # for dir path
START_DATA_HOUR_F3=$(date -d "$START_DATA_HOUR_F1" +"%Y%m%d%H")  # for sub version

$HADOOP fs -get $MODEL_BAK_PATH/last_model_bak.txt
last_model_version=$(cat last_model_bak.txt)
NUM=0
NUM_LIMIT=1
for ((i=0; i < $NUM_LIMIT; i++))
do
  echo "check $MDFS_PATH/$START_DATA_HOUR_F2/part-00000"
	$HADOOP fs -test -e $MDFS_PATH/$START_DATA_HOUR_F2/part-00000
	if [ $? -ne 0 ]; then
		continue
	fi
	let NUM++
  echo "download data start ..."
	$HADOOP fs -getmerge $MDFS_PATH/$START_DATA_HOUR_F2/part-* data/one_step_tmp/part-$NUM
# for test, pull total data too slow
#	$HADOOP fs -getmerge $MDFS_PATH/$START_DATA_HOUR_F2/part-0000* data/one_step_tmp/part-$NUM
done

if [ $NUM -lt $NUM_LIMIT ]; then
  echo "$NUM hours data ready, but request:$NUM_LIMIT"
	exit 0
fi

cat data/one_step_tmp/part-* > data/one_step_tmp/all_part
mv data/one_step_tmp/all_part data/one_step_tmp/allsample.txt
echo "download data, ret =" $?

echo "download feature_list"
$HADOOP fs -get $FEATURE_LIST_PATH/$START_DATA_HOUR_F2/fea_list.txt data/one_step_tmp/
echo "download fea_list, ret =" $?

# download model bak
echo "check model file: all_theta_dict.txt"
if [ ! -f data/all_theta_dict.txt ];then
  BAK_MODEL_FILE=$MODEL_BAK_PATH/model.$last_model_version
	$HADOOP fs -test -e $BAK_MODEL_FILE
	if [ $? -ne 0 ]; then
    echo "please upload model_bak: $BAK_MODEL_FILE"
    exit 1
	fi
  $HADOOP fs -get $BAK_MODEL_FILE data/all_theta_dict.txt
  ret=$?
  echo "get hadoop bak model, ret = $ret"
  if [ $ret -ne 0 ];then
    sendStr="$MODEL_VERSION get hadoop bak model failed"
    sh src/send.sh "$sendStr" "all"
    exit 1
  fi
fi
echo "download data end"

python src/theta_trans.py
echo "theta_trans, ret =" $?
python -u src/one_step_run.py $LR
ret=$?
if [ $ret -ne 0 ];then
  sendStr="$MODEL_VERSION-one_step_run-error"
  sh src/send.sh "$sendStr" "all"
  exit 1
fi
echo "one_step_run, ret = $ret"
python src/theta_trans2.py
echo "theta_trans2, ret =" $?

mv data/one_step_tmp/all_theta_dict.txt data/all_theta_dict.txt
echo "mv, ret =" $?
rm data/one_step_tmp/*
echo "rm one_step_tmp, ret =" $?

SUB_VERSION=$START_DATA_HOUR_F3
cp data/all_theta_dict.txt data/theta_dir/model.$SUB_VERSION
echo "cp model to theta_dir, ret =" $?
#python2 dcache/dcache_client.py $DCACHE_KEY $SUB_VERSION
python2 dcache/dcache_client.py $DCACHE_KEY $PART $SUB_VERSION
echo "push dcache, ret =" $?
$HADOOP fs -put -f data/theta_dir/model.$SUB_VERSION $MODEL_BAK_PATH
echo "push model to hdfs, ret =" $?

echo "change last_model_bak.txt << " $SUB_VERSION
echo $SUB_VERSION > last_model_bak.txt
$HADOOP fs -put -f last_model_bak.txt $MODEL_BAK_PATH/
echo "end"
