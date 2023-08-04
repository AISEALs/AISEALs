#/user/bin
echo "pwd" $(pwd)
echo "ls:" $(ls -l)
echo "----------------"
HOME_PATH=ES8_Venus
echo "set trainTime =" $1
echo $1 > $HOME_PATH/trainTime

MODEL_VERSION=12
PART=2
HADOOP_VERSION=es6
LR=4

if [ $# -ge 5 ];then
  MODEL_VERSION=$2
  PART=$3
  HADOOP_VERSION=$4
  LR=$5
  echo "MODEL_VERSION=" $MODEL_VERSION
  echo "PART=" $PART
  echo "HADOOP_VERSION=" $HADOOP_VERSION
  echo "LR=" $LR
fi
if [ $# -eq 6 ];then
  loadType=$6
  if [ $loadType -eq 1 ];then
  	echo "loadType=" $loadType
  	rm $HOME_PATH/data/all_theta_dict.txt
  	echo "train from hdfs, rm local model, ret =" $?
  else
  	echo "loadType=" $loadType
    echo "train from local"
  fi
fi

while true
do
	DATE_PATH=$(date -d "1 hour ago" +"%Y%m%d%H")
#	sh $HOME_PATH/run_hourly.sh $MODEL_VERSION $PART $HADOOP_VERSION $LR 2>&1 >>$HOME_PATH/trainLog/${DATE_PATH}.log
	sh $HOME_PATH/run_hourly.sh $MODEL_VERSION $PART $HADOOP_VERSION $LR
	sleep 60s
done