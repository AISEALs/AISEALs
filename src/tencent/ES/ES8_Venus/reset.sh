#/user/bin

#cd cpp & g++ es_noise.cpp -std=gnu++11

HADOOP=/data/home/hadoop-venus/bin/hadoop
MODEL_BAK_PATH=mdfs://cloudhdfs/pcfeeds/qbdata/nemoztwang/MiniVideo/Float/Praise/esModelBak/version
HOME_PATH=ES8_Venus
MODEL_VERSION=12
PART=2
DATE_NOW="`date +%Y%m%d%H`"

if [ $# -ge 2 ];then
  MODEL_VERSION=$1
  PART=$2
fi
rm $HOME_PATH/data/all_theta_dict.txt
echo "rm local model, ret =" $?
if [ $# -ge 3 ];then
  MODEL_TIME=$3
  REMOTE_PATH=$MODEL_BAK_PATH$MODEL_VERSION/model.$MODEL_TIME
  echo "reset MODEL_VERSION PART" $REMOTE_PATH $PART
  $HADOOP fs -get $REMOTE_PATH $HOME_PATH/data/all_theta_dict.txt
  echo "HADOOP fs -get, ret =" $?
else
  echo "reset MODEL_VERSION =" $MODEL_VERSION
  cp $HOME_PATH/data/test_theta.txt $HOME_PATH/data/all_theta_dict.txt
fi

#appname="run5.py"
#PROCESS=`ps -ef|grep $appname|grep -v grep|grep -v PPID|awk '{ print $2}'`
#for i in $PROCESS
#do
  #echo "Kill the $appname process [ $i ]"
  #kill -9 $i
#done

#rm $HOME_PATH/data/one_step_tmp/*


python2 $HOME_PATH/dcache/dcache_client.py $HOME_PATH $MODEL_VERSION $PART $DATE_NOW
echo "dcache_client, ret =" $?


REMOTE_PATH=$MODEL_BAK_PATH$MODEL_VERSION/model.$DATE_NOW
echo "put NEW_MODEl TO" $REMOTE_PATH 
$HADOOP fs -put $HOME_PATH/data/all_theta_dict.txt $REMOTE_PATH
echo "hadoop put model, ret =" $?


echo "end"

