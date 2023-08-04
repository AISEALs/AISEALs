#!/bin/bash

##*/2 * * * * cd /opt/dl/AISEALs/scripts/embeddings;./monitor_svr.sh /opt/dl/AISEALs/src/embeddings item2vec-realtime >> log.monitor 2>&1

current=`date "+%Y-%m-%d %H:%M:%S"`
echo "time: $current"
echo $#

task_path=$1
echo "task_path: $task_path"
task_name=$2
echo "task_name: $task_name"

cnt=`ps aux |grep $task_name | grep -v monitor_svr|grep -v grep |grep -v vi |wc -l`
echo "cnt: $cnt"

cd $task_path

if [ $cnt -eq 0 ]
then
  nohup /opt/soft/anaconda3/envs/pytorch/bin/python -u $task_name.py > /dev/null 2>&1 &
  echo $current" restart $task_name.py"
fi

