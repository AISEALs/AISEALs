#/user/bin

#cd cpp & g++ es_noise.cpp -std=gnu++11

HADOOP=/data/home/hadoop-venus/bin/hadoop
HADOOP_PRE=mdfs://cloudhdfs/pcfeeds/qbdata/nemoztwang/MiniVideo/Float/Praise
MODEL_BAK_PATH_PRE=mdfs://cloudhdfs/pcfeeds/qbdata/nemoztwang/MiniVideo/Float/Praise/esModelBak/version

MDFS_PATH=$HADOOP_PRE/es6
FEALIST_PATH=$HADOOP_PRE/es6_featureList
MODEL_BAK_PATH=${MODEL_BAK_PATH_PRE}12
MODEL_VERSION=12
HOME_PATH=ES8_Venus
PART=2
LR=4
if [ $# -eq 4 ];then
  MODEL_VERSION=$1
  PART=$2
  HADOOP_VERSION=$3
  LR=$4
  MDFS_PATH=$HADOOP_PRE/$HADOOP_VERSION
  FEALIST_PATH=$HADOOP_PRE/${HADOOP_VERSION}_featureList
  MODEL_BAK_PATH=$HADOOP_PRE/esModelBak/version$MODEL_VERSION
fi

DATE_PATH=$(date -d "1 hour ago" +"%Y-%m-%d/%H")
echo "DATE_PATH=" $DATE_PATH 
date_model=$(date -d "1 hour ago" +"%Y%m%d%H")
DATE1=${DATE_PATH:0:10}
DATE2=${DATE_PATH:11}
DATE="$DATE1 $DATE2"
dataline=$(cat $HOME_PATH/trainTime)
datetimestamp="${dataline:0:10} ${dataline:11}"
cur_dateTime="`date +%Y-%m-%d,%H:%M`"
echo $cur_dateTime "#########################"
echo "nowTime" $DATE_PATH
echo "trainTime" $dataline
timestamp_now=$(date -d "$DATE" +"%s")
timestamp_trained=$(date -d "$datetimestamp" +"%s")
if [ $timestamp_now -gt $timestamp_trained ]; then
    echo "last hour no train"
    sendStr=$MODEL_VERSION",last_hour_no_train,neededTrainTime="$dataline",nowTime="$DATE_PATH
    sh $HOME_PATH/send.sh $sendStr "all"
elif [ $timestamp_now -eq $timestamp_trained ]; then
    echo "start train"
else
    echo "has trained"
    exit 0
fi

hh=$DATE2
if [ $hh -le 7 ] && [ $hh -ge 1 ]; then
	exit 0
fi
echo "download data start..."
NUM=0
NUM_LIMIT=1
for i in $( seq 1 $NUM_LIMIT )
do
	#DATE_PATH=$(date -d "$i hour ago" +"%Y-%m-%d/%H")
   	echo "download: $DATE_PATH"
	$HADOOP fs -test -e $MDFS_PATH/$DATE_PATH/part-00000 
	#2>$HOME_PATH/hadoopErr.log
	if [ $? -ne 0 ]; then 
		continue
	fi
	let NUM++
	$HADOOP fs -getmerge $MDFS_PATH/$DATE_PATH/part-* $HOME_PATH/data/one_step_tmp/part-$NUM 
	#2>$HOME_PATH/hadoopErr.log
        #$HADOOP fs -get $FEALIST_PATH/$DATE_PATH/fea_list.txt $HOME_PATH/data/one_step_tmp/
done
if [ $NUM -eq 0 ]; then
  echo "data no ready"
	exit 0
fi
cat $HOME_PATH/data/one_step_tmp/part-* > $HOME_PATH/data/one_step_tmp/all_part
mv $HOME_PATH/data/one_step_tmp/all_part $HOME_PATH/data/one_step_tmp/allsample.txt
$HADOOP fs -get $FEALIST_PATH/$DATE_PATH/fea_list.txt $HOME_PATH/data/one_step_tmp/
echo "download fea_list, ret =" $?
#2>$HOME_PATH/hadoopErr.log
echo "check model file: all_theta_dict.txt"
if [ ! -f $HOME_PATH/data/all_theta_dict.txt ];then
  MODEL_TIME1=$(date -d "$DATE 1 hour ago" +"%Y%m%d")
  MODEL_TIME2=$(date -d "$DATE 1 hour ago" +"%H")
  if [ $MODEL_TIME2 -le 7 ] && [ $MODEL_TIME2 -ge 1 ]; then
      MODEL_TIME2="00"
  fi
  MODEL_TIME=$MODEL_TIME1$MODEL_TIME2
  MODEL_PATH=$MODEL_BAK_PATH/model.$MODEL_TIME
  echo "get hadoop model from" $MODEL_PATH
  $HADOOP fs -get $MODEL_PATH $HOME_PATH/data/all_theta_dict.txt
  echo "get hadoop model, ret =" $? 
else
  echo "local model exist"
fi
echo "download end, update starting"
#exit 0

#python $HOME_PATH/src/pre_process.py $HOME_PATH $MODEL_VERSION
#rm $HOME_PATH/data/one_step_tmp/part-00000
python $HOME_PATH/src/theta_trans.py $HOME_PATH
echo "theta_trans, ret =" $?
python $HOME_PATH/src/one_step_run.py $HOME_PATH $LR
echo "one_step_run, ret =" $?
if [ $? -ne 0 ];then
  sendStr=$MODEL_VERSION",one_step_run_error,neededTrainTime="$dataline
  sh $HOME_PATH/send.sh $sendStr "all"
  exit 1
fi
python $HOME_PATH/src/theta_trans2.py $HOME_PATH
echo "theta_trans2, ret =" $?
mv $HOME_PATH/data/all_theta_dict.txt $HOME_PATH/data/all_theta_dict.txt.bak
echo "mv1, ret =" $?
mv $HOME_PATH/data/one_step_tmp/all_theta_dict.txt $HOME_PATH/data/all_theta_dict.txt
echo "mv2, ret =" $?
rm $HOME_PATH/data/one_step_tmp/*
echo "rm one_step_tmp, ret =" $?
cp $HOME_PATH/data/all_theta_dict.txt $HOME_PATH/data/theta_dir/model.$date_model
echo "cp model to theta_dir, ret =" $?
python2 $HOME_PATH/dcache/dcache_client.py $HOME_PATH $MODEL_VERSION $PART $date_model
echo "push dcache, ret =" $?
$HADOOP fs -put $HOME_PATH/data/theta_dir/model.$date_model $MODEL_BAK_PATH
echo "push model to hdfs, ret =" $?
DATE_NEXT=$(date -d "$DATE 1 hour" +"%Y-%m-%d/%H")
DATE1=${DATE_NEXT:0:10}
DATE2=${DATE_NEXT:11}
if [ $DATE2 -le 7 ] && [ $DATE2 -ge 1 ]; then
      DATE2="08"
fi
DATE_NEED="$DATE1/$DATE2"
echo "change trainTime" $DATE_NEED
echo $DATE_NEED > $HOME_PATH/trainTime
echo "end"