export DATA_DIR=/cfs/cfs-lq0xu8jj/wechat2022_data/albef_data

echo '-------------------------------------------------------'
ls
echo '-------------------------------------------------------'
cd ALBEF || exit 1
rm -rf data
rm -rf output
ln -s ${DATA_DIR} data
#tree ./data/
echo '-------------------------------------------------------'
echo "nvidia-sim:"
nvidia-smi
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1 #cuda11必备
# shellcheck disable=SC2068
echo $@
echo 'start train:'
# $@ 获取shell脚本的入参，当用户选择集群模式时，平台会在shell脚本后面自动拼上nproc_per_node、nnodes、node_rank、master_addr、master_port等参数
# shellcheck disable=SC2068
python -m torch.distributed.launch $@ --use_env Retrieval.py \
--config ./configs/Retrieval_flickr.yaml \
--output_dir output/Retrieval_flickr \
--checkpoint ./data/ALBEF.pth

#python Retrieval.py \
#--config ./configs/Retrieval_flickr.yaml \
#--output_dir ./data/output/Retrieval_flickr \
#--checkpoint ./data/ALBEF.pth
