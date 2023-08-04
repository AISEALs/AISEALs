#/user/bin

#cd cpp & g++ es_noise.cpp -std=gnu++11

HADOOP=/data/home/hadoop-venus/bin/hadoop
MODEL_BAK_PATH=mdfs://cloudhdfs/pcfeeds/qbdata/nemoztwang/MiniVideo/Float/Praise/esModelBak/version
HOME_PATH=ES8_Venus
MODEL_VERSION=12
PART=2

if [ $# -eq 2 ];then
  MODEL_VERSION=$1
  MODEL_TIME=$2
  REMOTE_PATH=$MODEL_BAK_PATH$MODEL_VERSION/model.$MODEL_TIME
  $HADOOP fs -put $HOME_PATH/data/all_theta_dict.txt $REMOTE_PATH
  echo "hadoop put local model, ret =" $?
fi

if [ $# -eq 3 ];then
  MODEL_VERSION=$1
  MODEL_TIME=$2
  NEW_TIME=$3
  REMOTE_PATH=$MODEL_BAK_PATH$MODEL_VERSION/model.$MODEL_TIME
  REMOTE_PATH2=$MODEL_BAK_PATH$MODEL_VERSION/model.$NEW_TIME
  echo "REMOTE_PATH" $REMOTE_PATH
  echo "REMOTE_PATH2" $REMOTE_PATH2
  $HADOOP fs -cp $REMOTE_PATH $REMOTE_PATH2
  echo "hadoop put remote model, ret =" $?
fi

echo "end"

