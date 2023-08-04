#nohup python finnal_predict_labels.py > /dev/null 2>&1 &
#nohup python -u finnal_predict_labels.py > log/nohup.log 2>&1 &
nohup /opt/soft/anaconda3/envs/tensorflow/bin/python finnal_predict_labels.py > log/nohup.log 2>&1 &
