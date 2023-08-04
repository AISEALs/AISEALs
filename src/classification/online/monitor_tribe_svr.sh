#!/bin/bash

cnt=`ps aux |grep finnal_predict_labels.py |grep -v grep |grep -v vi |wc -l`

current=`date "+%Y-%m-%d %H:%M:%S"`
cd /opt/tribe_labels/online

if [ $cnt -eq 0 ]
then
  nohup /opt/soft/anaconda3/envs/tensorflow/bin/python -u finnal_predict_labels.py > log/nohup.log 2>&1 &
  echo $current" restart finnal_predict_labels.py "
fi

