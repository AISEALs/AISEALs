#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../")
import numerous
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from numerous.optimizers.optimizer_context import OptimizerContext
from numerous.optimizers.adam import Adam
from numerous.distributions.uniform import Uniform
from numerous.distributions.normal import Normal
from numerous.client.config import get_config
from mmoe_model import mmoe_model_fn_rerank
import autoIntModel
import deepModel
'''
去掉基础模型分，验证可知线上作用很低，需要bia塔进行加强
'''

def userLog(loginfo):
    print("USERLOG->%s" % loginfo)


# task
numerous.training.Task(model_name="qbReRank", worker_thread=4, worker_async_thread=2, server_thread=4)

# saver
numerous.training.Saver(expire_time=864000, dump_interval="dir:1")

# reader
sample_format = numerous.reader.SampleFormat(sample_col_delimiter="|",
                                             label_col_pos=1,
                                             feature_col_pos=2,
                                             sample_feature_delimiter=";",
                                             feature_section_delimiter=":",
                                             key_section_pos=0,
                                             slot_section_pos=1,
                                             value_section_pos=2)

# reader mode
reader = numerous.reader.Reader()
reader._file_mini_batch_size = -1

model_base_name = get_config().get_option("model_base_name", default_value="")
model_base_version = get_config().get_option("model_base_version", default_value="")
model_name = get_config().get_option("model_name", default_value="")
playratio_threshold = float(get_config().get_option("userdefine.config.playratio_threshold", default_value = "1.0")) 
ratio_use_denoise = bool(int(get_config().get_option("userdefine.config.ratio_use_denoise", default_value = "0")))
time_use_denoise =  bool(int(get_config().get_option("userdefine.config.time_use_denoise", default_value = "0")))
print('model_name = %s, model_base_name = %s, model_base_version = %s' % (
model_name, model_base_name, model_base_version))
if model_base_version != '' and model_base_name != model_name:
    inherit_config = numerous.training.ModelInheritConfig(model_base_version)
    inherit_config.exclude_dense_params()
    numerous.training.Recover(model_inherit_config=inherit_config)
numerous.framework.HotKeyGenerationConfig(dump_hot_embedding_gb=2.0)

# optimizer
OptimizerContext().set_sparse_optimizer(optimizer=Adam(rho1=0.9, rho2=0.999, eps=0.001, time_thresh=5000))
OptimizerContext().set_dense_optimizer(optimizer=Adam(rho1=0.9, rho2=0.999, eps=0.00005, time_thresh=5000))

# parameter initialize
sparse_initializer = Uniform(left=-0.001, right=0.001)
normal = Normal(mu=0.0, lam=0.1)

# control placeholder
is_training = tf.placeholder(tf.bool, name='training')
numerous.reader.ControlPlaceholder(is_training, training_phase=True, inference_phase=False)
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
numerous.reader.ControlPlaceholder(keep_prob, training_phase=0.8, inference_phase=1.0)

# 点击历史个数
NUM_IN_FLUSH = 5
EMBEDDING_DIM = 16
sem_vec_dim = 32

# label 定义
ITIME_LABEL_SLOT = 900
PLAY_TIME_LABEL_SLOT = 1000
POS_LABEL_SLOT = 901
PR_LABEL_SLOT = 902
FI_LABEL_SLOT = 903
SK_LABEL_SLOT = 904
PT_LABEL_SLOT = 905
POS_PROBALILY_SLOT = 906
# SEM_VEC_SLOT = 8001

item_feature = [1001, 2001, 3001, 4001, 7101, 7201, 7301, 7601]
user_feature = [5001, 6001, 7001]
double_hash_slots = [7401, 7501]
num_item_feature = len(item_feature)
num_user_feature = len(user_feature)

item_feature_table = []
user_feature_table = []

# item feature
for start_slot_id in item_feature:
    item_emb = []
    double_hash_config = numerous.framework.DoubleHashingComperssConfig(hash_range=(1 << 26),
                                                                        embedding_combine_method="sum")
    for m_slot_id in range(start_slot_id, start_slot_id + NUM_IN_FLUSH):
        if start_slot_id in double_hash_slots:
            fuse_out = numerous.framework.SparseEmbedding(
                table_name="sparse_w_" + str(m_slot_id),
                model_slice="embedding",
                embedding_dim=EMBEDDING_DIM,
                save_type="int8_float32",
                optimizer=Adam(rho1=0.9, rho2=0.999, eps=0.001, time_thresh=5000),
                distribution=sparse_initializer,
                dump_opt_args=True,
                slot_ids=[str(m_slot_id)],
                combine_method="sum",
                double_hashing_compress_config=double_hash_config)
        else:
            fuse_out = numerous.framework.SparseEmbedding(
                table_name="sparse_w_" + str(m_slot_id),
                model_slice="embedding",
                embedding_dim=EMBEDDING_DIM,
                save_type="int8_float32",
                optimizer=Adam(rho1=0.9, rho2=0.999, eps=0.001, time_thresh=5000),
                distribution=sparse_initializer,
                dump_opt_args=True,
                slot_ids=[str(m_slot_id)],
                combine_method="sum")
        item_emb.append(fuse_out[0])
    item_feature_table.append(item_emb)

# user feature
for user_feature_slot in user_feature:
    fuse_out = numerous.framework.SparseEmbedding(
        table_name="sparse_w_" + str(user_feature_slot),
        model_slice="embedding",
        embedding_dim=EMBEDDING_DIM,
        save_type="int8_float32",
        optimizer=Adam(rho1=0.9, rho2=0.999, eps=0.001, time_thresh=5000),
        distribution=sparse_initializer,
        dump_opt_args=True,
        slot_ids=[str(user_feature_slot)],
        combine_method="sum")
    user_feature_table.append(fuse_out[0])
user_feature = tf.concat(user_feature_table, axis=1, name="user_feature_table")

# 基础模型打分特征
dense_slots = [9001, 9101, 9201, 9301]
dense_feas = []
for idx in range(0, NUM_IN_FLUSH):
    single_dense_fea = []
    for start_slot in dense_slots:
        curr_slot_id = start_slot + idx
        print("USERLOG:slotid:%d" % curr_slot_id)
        dense_pl = tf.placeholder(tf.float32, shape=[None, 1], name="dense_pl_%d" % curr_slot_id)
        numerous.reader.DensePlaceholder(dense_pl, slot_id=str(curr_slot_id), lower_bound=1, upper_bound=1, )
        single_dense_fea.append(dense_pl)
    singe_dense_scores = tf.concat(single_dense_fea, axis=1, name="dense_score_fea_%d" % idx)
    print("USERLOG:singe_dense_scoresSize:", singe_dense_scores.get_shape())
    dense_feas.append(tf.clip_by_value(singe_dense_scores, 0, 1))

# 语义索引特征
'''
sem_feature_table = []
for sem_slot_id in range(SEM_VEC_SLOT, SEM_VEC_SLOT + NUM_IN_FLUSH):
    sem_embedding = tf.placeholder(tf.float32, shape=[None, sem_vec_dim],
                                   name="sem_embedding_" + str(sem_slot_id))
    numerous.reader.DensePlaceholder(
        sem_embedding,
        slot_id=str(sem_slot_id),
        lower_bound=20001,
        upper_bound=20001 + sem_vec_dim - 1,
    )
    sem_feature_table.append(sem_embedding)
'''

# 位置特征
dense_fea_npos = tf.placeholder(tf.float32, shape=[None, NUM_IN_FLUSH],
                                name="pos_label")
numerous.reader.DensePlaceholder(dense_fea_npos, slot_id=str(POS_LABEL_SLOT),
                                 lower_bound=20001,
                                 upper_bound=20001 + NUM_IN_FLUSH - 1)
fea_npos = tf.split(dense_fea_npos, 5, axis=1)

# 基础模型pr输出label
dense_fea_ratio = tf.placeholder(tf.float32, shape=[None, NUM_IN_FLUSH],
                                 name="ratio_label")
numerous.reader.DensePlaceholder(dense_fea_ratio, slot_id=str(PR_LABEL_SLOT),
                                 lower_bound=20001,
                                 upper_bound=20001 + NUM_IN_FLUSH - 1)
# 基础模型pt输出label
dense_fea_time = tf.placeholder(tf.float32, shape=[None, NUM_IN_FLUSH],
                                 name="time_label")
numerous.reader.DensePlaceholder(dense_fea_time, slot_id=str(PT_LABEL_SLOT),
                                 lower_bound=20001,
                                 upper_bound=20001 + NUM_IN_FLUSH - 1)
tasks_base_pred = [dense_fea_ratio, dense_fea_time]
# 序列位置采样
pos_ratio = tf.placeholder(tf.float32, shape=[None, NUM_IN_FLUSH],
                                 name="pos_ratio")
numerous.reader.DensePlaceholder(pos_ratio, slot_id=str(POS_PROBALILY_SLOT),
                                 lower_bound=20001,
                                 upper_bound=20001 + NUM_IN_FLUSH - 1)

# 特征concat
att_slot_embedding_vector = []
for slot_id in range(0, NUM_IN_FLUSH):
    m_vlaues = []
    # item特征
    for feature_id in range(0, num_item_feature):
        m_vlaues.append(item_feature_table[feature_id][slot_id])
    # 用户特征
    m_vlaues.append(user_feature)
    # 语义特征
    # m_vlaues.append(sem_feature_table[slot_id])
    # 位置特征
    m_vlaues.append(fea_npos[slot_id])
    # 基础模型打分特征
    # m_vlaues.append(dense_feas[slot_id])

    m_one_item_concat = tf.concat(m_vlaues, axis=1, name="attention_concat_" + str(slot_id))
    att_slot_embedding_vector.append(m_one_item_concat)

all_item_concat = tf.concat(att_slot_embedding_vector, axis=1, name="attention_concat_all")

# 单个item
size_of_concat_emb = EMBEDDING_DIM * (num_item_feature + num_user_feature) + 1
print("ALLDIM:", all_item_concat.get_shape())
embedding_attention_layer = tf.reshape(all_item_concat,
                                       shape=[tf.shape(all_item_concat)[0], NUM_IN_FLUSH, size_of_concat_emb])
print("ALLDIM2:", embedding_attention_layer.get_shape())

#  embedding_attention_layer   b * 5 * size_of_concat_emb
tasks_name = ["ratio", "time"]
tasks_pred = []
tasks_logit = []
taske_base_pred = [dense_fea_ratio, dense_fea_time]
layersizes = [size_of_concat_emb*2, size_of_concat_emb]
num_experts = 4
num_tasks = len(tasks_name)
tower_outputs = mmoe_model_fn_rerank(embedding_attention_layer, layersizes, num_experts, num_tasks, "rerank",True)
for i in range(num_tasks):
  #embedding_attention_layer = tower_outputs[i]

  # AutoInt
  num_heads = 1
  num_blocks = 1
  attention_embedding_size = size_of_concat_emb / num_heads
  model = autoIntModel.Model(input=tower_outputs[i],
                             attention_embedding_size=attention_embedding_size,
                             num_heads=num_heads,
                             num_blocks=num_blocks,
                             keep_prob=keep_prob,
                             name=tasks_name[i])
  attention_output = model.run()
  m_attention_num = tf.split(attention_output, num_or_size_splits=NUM_IN_FLUSH, axis=1)
  print("SHAPE:"+tasks_name[i], m_attention_num[0].get_shape().as_list())

  layer_size = [attention_embedding_size * num_heads * num_blocks * 2,
                attention_embedding_size * num_heads * num_blocks,
                512,
                1]

  all_output_result = []
  for item_id in range(0, NUM_IN_FLUSH):
      layer_1 = tf.layers.dense(inputs=m_attention_num[item_id],
                                units=layer_size[0],
                                activation=tf.nn.selu,
                                name=tasks_name[i]+"_layer_1_" + str(item_id))
      print("SHAPE2:", layer_1.get_shape().as_list())
      layer_2 = tf.layers.dense(inputs=layer_1,
                                units=layer_size[1],
                                activation=tf.nn.selu,
                                name=tasks_name[i]+"_layer_2_" + str(item_id))
      print("SHAPE3:", layer_2.get_shape().as_list())
      res_layer = layer_2 + m_attention_num[item_id]
      print("SHAPE4:", res_layer.get_shape().as_list())
      layer_3 = tf.layers.dense(inputs=res_layer,
                                units=layer_size[2],
                                activation=tf.nn.selu,
                                name=tasks_name[i]+"_layer_3_" + str(item_id))

      out_layer = tf.layers.dense(inputs=layer_3,
                                  units=layer_size[3],
                                  activation=None,
                                  name=tasks_name[i]+"_output_layer" + str(item_id))

      all_output_result.append(out_layer)

  out_loss_concat = tf.concat(all_output_result, axis=1, name=tasks_name[i]+"_out_loss_concat")
  tasks_logit.append(out_loss_concat)
  y_pred = tf.sigmoid(out_loss_concat, name=tasks_name[i]+"_pred_sigmoid")
  tasks_pred.append(y_pred)


# 训练用label
playtime_label = tf.placeholder(tf.float32, shape=[None, NUM_IN_FLUSH],
                                name="playtime_label")
numerous.reader.DensePlaceholder(playtime_label, slot_id=str(PLAY_TIME_LABEL_SLOT),
                                 lower_bound=20001,
                                 upper_bound=20001 + NUM_IN_FLUSH - 1)

iTime_label = tf.placeholder(tf.float32, shape=[None, NUM_IN_FLUSH],
                             name="iTime_label")
numerous.reader.DensePlaceholder(iTime_label, slot_id=str(ITIME_LABEL_SLOT),
                                 lower_bound=20001,
                                 upper_bound=20001 + NUM_IN_FLUSH - 1)

final_label = tf.clip_by_value(playtime_label / (iTime_label*playratio_threshold + 1e-8), 0., 1.)

time_label =  tf.clip_by_value(playtime_label / 150, 0., 1.)
tasks_label = [final_label, time_label]

loss_all = 0
# denoise
DenoiseNet = deepModel.DenoiseNet()
pos_probality_weight = pos_ratio
compare_mse_list = []
rerank_mse_list = []
rerank_loss_list = []
rerank_denoise_loss_list = []
tasks_use_denoise = [ratio_use_denoise, time_use_denoise]
print("ratio_use_denoise", ratio_use_denoise)
print("time_use_denoise", time_use_denoise)
for i in range(num_tasks):
  denoise_net_ratio_weight = tf.ones_like(tasks_label[i])
  if tasks_use_denoise[i]:
    denoise_net_ratio_logit \
        = DenoiseNet.get_denoise_net(all_item_concat, NUM_IN_FLUSH, is_training, tasks_name[i])
    noise_sample_flag = tf.where(tf.abs(playtime_label - iTime_label) < 0.01, tf.fill(tf.shape(playtime_label), 1.),
                                 tf.fill(tf.shape(playtime_label), 0.))
    weightnet_sample_mask = noise_sample_flag < 0.5

    # ratio denoise net
    denoise_net_ratio_label = tasks_label[i] 
    denoise_net_ratio_label = tf.boolean_mask(denoise_net_ratio_label, weightnet_sample_mask)
    denoise_net_ratio_logit_mask = tf.boolean_mask(denoise_net_ratio_logit, weightnet_sample_mask)

    denoise_net_ratio_loss = tf.cond(tf.equal(tf.size(denoise_net_ratio_label), 0), \
        lambda : tf.reduce_mean(tf.zeros_like(denoise_net_ratio_label)), \
        lambda : tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=denoise_net_ratio_logit_mask, labels=denoise_net_ratio_label)))
    if tasks_name[i] == "time":
      weightnet_pred_weight = denoise_net_ratio_logit
      weightnet_label = tasks_label[i]

      weightnet_pred_diff = tf.abs(tf.subtract(weightnet_pred_weight, weightnet_label))
      weightnet_pred_diff = tf.clip_by_value(weightnet_pred_diff, 1e-8,
                                             1.0) * 10  # 放大一定倍数， diff 2s = 0.02 ， 0.02 *10 = 0.2 变换后 接近1
      weightnet_pred_diff1 = tf.sigmoid(
          tf.divide(tf.ones_like(weightnet_pred_diff, tf.float32), weightnet_pred_diff))
      weightnet_pred_weight = (weightnet_pred_diff1 - 0.5) * 2
      denoise_net_ratio_weight = tf.stop_gradient(weightnet_pred_weight)
    else:
      denoise_net_ratio_pred = tf.sigmoid(denoise_net_ratio_logit, "denoise_net_ratio_pred")
      denoise_net_ratio_weight = tf.stop_gradient(denoise_net_ratio_pred, name = "denoise_net_ratio_weight1")
      denoise_net_ratio_weight = tf.where(weightnet_sample_mask, tf.ones_like(denoise_net_ratio_weight, tf.float32), denoise_net_ratio_weight, name = "denoise_net_ratio_weight2")
  #denoise_weights.append(denoise_net_ratio_weight)
  

  label = tasks_label[i]
  base_pred = tasks_base_pred[i]
  if tasks_name[i] == "ratio":
    base_pred = tf.clip_by_value(base_pred/playratio_threshold,0.,1.)
  # 基础模型loss，debug用
  compare_mse = tf.reduce_mean(tf.squared_difference(label, base_pred), name="basic_model_mse_"+tasks_name[i])
  rerank_mse = tf.reduce_mean(tf.squared_difference(label, tasks_pred[i]), name="rarank_model_mse_"+tasks_name[i])
  compare_mse_list.append(compare_mse)
  rerank_mse_list.append(rerank_mse)
  ce_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(label, tasks_logit[i], denoise_net_ratio_weight) * pos_probality_weight)
  loss_all += ce_loss
  rerank_loss_list.append(ce_loss)
  if tasks_use_denoise[i]:
    loss_all += denoise_net_ratio_loss
    rerank_denoise_loss_list.append(denoise_net_ratio_loss)

# estimator
target = numerous.optimizers.OptimizerContext().minimize(loss_all)
target.set_fetches([loss_all])

estimator = numerous.estimator.Estimator()
ctx = numerous.get_default_context()
train_hook = numerous.estimator.LegacyTrainDataHook()
eval_hook = numerous.estimator.LegacyEvalDataHook()

y_pred_all = tf.concat(tasks_pred, axis=1, name="pred_sigmoid")

# summary_hook
tensorboard_path = numerous.client.config.get_config().get_option("userdefine.config.tensorboard_path",
                                                                  default_value="./log")
print("[DEBUG] tensorboard_path: {0}".format(tensorboard_path))
summary_hook = numerous.estimator.SummaryHook(tensorboard_path, skip_step=0)
summary_hook.add_summary(tf.summary.scalar("loss_all", loss_all))

# metric hook
metric_hook = numerous.estimator.MetricHook(report_every_n_batch=1)
#metric_hook.add_metric(numerous.metrics.DistributeAuc(label_list, pred_list, name="auc"))
metric_hook.add_metric(numerous.metrics.DistributeMean(loss_all, name="loss_all"))
for i in range(num_tasks):
  metric_hook.add_metric(numerous.metrics.DistributeMean(compare_mse_list[i], name="compare_mse_"+tasks_name[i]))
  metric_hook.add_metric(numerous.metrics.DistributeMean(rerank_mse_list[i], name="rerank_mse_"+tasks_name[i]))
  metric_hook.add_metric(numerous.metrics.DistributeMean(rerank_loss_list[i], name="loss_"+tasks_name[i]))
  if tasks_use_denoise[i]:
    metric_hook.add_metric(numerous.metrics.DistributeMean(rerank_denoise_loss_list[i], name="denoise_loss"+tasks_name[i]))
  
# start numerous
if ctx.numerous_reader.is_stream():
    estimator.train(target=target, hooks=[train_hook, metric_hook])
    estimator.close()
else:
    for e in range(ctx.numerous_reader.epoch()):
        target.set_epoch(e)
        estimator.train(target=target, hooks=[train_hook, metric_hook, summary_hook])
    estimator.close()
