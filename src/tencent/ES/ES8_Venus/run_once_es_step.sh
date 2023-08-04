#/user/bin

#cd cpp & g++ es_noise.cpp -std=gnu++11

HADOOP=/data/home/carliu/hadoop-mdfs/bin/hadoop
MDFS_PATH=mdfs://cloudhdfs/pcfeeds/qbdata/nemoztwang/MiniVideo/Float/Praise/es5
HOME_PATH=/data/home/yimshi/ES6
MODEL_VERSION=15

# DATE_PATH=$(date -d "1 hour ago" +"%Y-%m-%d/%H")
echo $DATE_PATH
# hh=`date '+%H'`
DATE_PATH="2021-12-18/11"
hh=12
if [ $hh -le 7 ] && [ $hh -ge 1 ]; then
	exit 0
fi
rm $HOME_PATH/data/one_step_tmp/*

echo "download data start..."
NUM=0
NUM_LIMIT=1
for i in $( seq 1 $NUM_LIMIT )
do
	#DATE_PATH=$(date -d "$i hour ago" +"%Y-%m-%d/%H")
   	echo $DATE_PATH
	$HADOOP fs -test -e $MDFS_PATH/$DATE_PATH/part-00000
	if [ $? -ne 0 ]; then 
		continue
	fi
	let NUM++
	$HADOOP fs -get $MDFS_PATH/$DATE_PATH/part-00000 $HOME_PATH/data/one_step_tmp/part-$NUM
done
if [ $NUM -eq 0 ]; then
	"no data"
	exit 0
fi
cat $HOME_PATH/data/one_step_tmp/part-* > $HOME_PATH/data/one_step_tmp/all_part
mv $HOME_PATH/data/one_step_tmp/all_part $HOME_PATH/data/one_step_tmp/part-00000

echo "download end, update starting"
#exit 0

python $HOME_PATH/src/pre_process.py $HOME_PATH $MODEL_VERSION
rm $HOME_PATH/data/one_step_tmp/part-00000
python $HOME_PATH/src/theta_trans.py $HOME_PATH
python $HOME_PATH/src/one_step_run.py $HOME_PATH
python $HOME_PATH/src/theta_trans2.py $HOME_PATH
#mv $HOME_PATH/data/all_theta_dict.txt $HOME_PATH/data/all_theta_dict.txt.bak
#mv $HOME_PATH/data/one_step_tmp/all_theta_dict.txt $HOME_PATH/data/all_theta_dict.txt
# cat $HOME_PATH/data/all_theta_dict.txt | grep -v "e-" | sort -k2,2f > $HOME_PATH/data/k
python $HOME_PATH/dcache/dcache_client.py $HOME_PATH $MODEL_VERSION
echo "end"
