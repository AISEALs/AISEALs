#export MODEL_BASE_DIR=../model/cnn_text_classification/output
export DATA_DIR=/cfs/cfs-lq0xu8jj/wechat2022_data/data
export SAVE_DIR=/cfs/cfs-lq0xu8jj/wechat2022_data/data/save
export CACHE_DIR=/cfs/cfs-lq0xu8jj/wechat2022_data/data/cache

echo '---------'
ls
echo '---------'
cd wechat2022 || exit 1
rm -rf data
ln -s ${DATA_DIR} data
tree ./data/
sudo chmod 777 -R ${SAVE_DIR}
sudo chmod 777 -R ${CACHE_DIR}
tree ./save
echo '---------'
ls
echo 'start train:'
args="$@"
for var in "$args":
do
    echo $var
done

python inference.py $args