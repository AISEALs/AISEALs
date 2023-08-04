#!/usr/bin/python3

import utils
from config.config import get_gpu_config, get_common_args
from layers import MMoEModel, DenoiseLayer, HadamardLayer, AutoIntModel, WatchTimeGainUpdater
from layers import AutoGroupLayer
from config import slots_base as slots
from model_zoo.models import NumerousModel
from model_zoo.inputs import Inputs
import numerous
import tensorflow as tf
from numerous.train.run_config import RunConfig
from numerous.utils.config import get_config
import os

os.environ["NCCL_LAUNCH_MODE"] = "GROUP"

print("eager model:", tf.executing_eagerly())
# import subprocess
# subprocess.call(["sudo", "/usr/bin/python3", "-m", "pip", "install", "numerous-modelzoo"])

denoise_flag_check = int(get_config().get_option("userdefine.config.denoise_flag_check", default_value="0"))
denoise_flag_check = bool(denoise_flag_check)
autogroup_order_num_list = get_config().get_option("userdefine.config.autogroup_order_num_list",
                                                   default_value="2:100_3:200")
temperature = float(get_config().get_option("userdefine.config.temperature", default_value="1.0"))
biz_debias = get_config().get_option("userdefine.config.biz_debias", default_value="3:1.2_4,5:1.2")
duration_bin_weights = get_config().get_option("userdefine.config.duration_bins_weight", default_value="1:1.0")

print("denoise_flag_check: {}".format(denoise_flag_check))
print("autogroup_order_num_list: {}".format(autogroup_order_num_list))
print("temperature: {}".format(temperature))
print('biz_debias: {}'.format(biz_debias))

args = get_common_args()
GPU_CONFIG = get_gpu_config()
old_feasign_slots, new_feasign_slots = slots.get_dnn_slots()
lr_slots = slots.get_wide_slots()
invalid_slots, filter_slots = slots.get_filter_slots()
hash_slots = slots.get_hash_slots()
hadamard_slot = slots.get_hadamard_slot()

replaced_slot = 1942

# attention history slots
item_start_slot, num_click_history = 1541, 30
item_end_slot = item_start_slot + num_click_history - 1
attention_item_slots = utils.parseSlotsList(f'{item_start_slot}-{item_end_slot}')
attention_slot_bias = [0, 90, 60, 30, 120]  # [item, pos, sub, tag, playtime]
# [float_click_ids, interval_time_bin, sub_list, tag_list, playtime_list]
# [target_item, target_pos, target_sub, target_tag, target_playtime, target_vtime]
target_slots = [600, 625, 697, 1832, 624]
seq_names = ['item', 'pos', 'sub', 'tag', 'playtime']

dnn_slots = old_feasign_slots + new_feasign_slots + hash_slots
dnn_feature_names = [f'sparse_w_{slot_id}' if slot_id not in filter_slots else f'sparse_w_{replaced_slot}'
                     for slot_id in dnn_slots if slot_id not in invalid_slots]
lr_feature_slots = ','.join([str(slot_id) for slot_id in lr_slots if slot_id not in invalid_slots])
print(f'dnn_slots size:{len(dnn_slots)}, lr_slots size:{len(lr_slots)}')

# autogroup
autogroup_slots = [slot_id for slot_id in dnn_slots if slot_id not in target_slots + hash_slots]
autogroup_feature_names = [f'sparse_w_{slot_id}' if slot_id not in filter_slots else f'sparse_w_{replaced_slot}'
                           for slot_id in autogroup_slots if slot_id not in invalid_slots]
autogroup_slot_num = len(autogroup_feature_names) + len(hadamard_slot)

ORDERS = []
GROUP_NUMS = []
for order_num in autogroup_order_num_list.strip().split('_'):
    sp = order_num.strip().split(':')
    ORDERS.append(int(sp[0]))
    GROUP_NUMS.append(int(sp[1]))

embedding_dim = 8
num_tasks = 3
task_names = ['ratio', 'skip', 'finish', 'time']
wtg_tasks = ["finish"]

label_configs = [['stime', 10000, 1, 1],
                 ['splaytime', 10000, 2, 2],
                 ['ispraise', 10000, 9, 9],
                 ['isshare', 10000, 10, 10],
                 ['isreadcomment', 10000, 11, 11],
                 ['iswritecomment', 10000, 12, 12],
                 ['isfollow', 10000, 13, 13],
                 ['isfeedback', 10000, 14, 14],
                 ]

biz_slot = 2104
biz_weights = []
for x in biz_debias.strip().split('_'):
    sp = x.split(':')
    for biz in sp[0].split(','):
        biz_weights.append((int(biz), float(sp[1])))
print('biz_weights:', biz_weights)

duration_configs = [['duration', 3731, 1, 1]]
duration_boundaries = [7, 9, 11, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300, 400, 500]
duration_weights = [1.0] * (len(duration_boundaries)+1)
print('duration_weights:\n', duration_boundaries)
for bin_weight in duration_bin_weights.split('|'):
    bin_w = bin_weight.split(':')
    bin_idx = int(bin_w[0])
    w = float(bin_w[1])
    duration_weights[bin_idx] = w
    start = duration_boundaries[bin_idx - 1] if bin_idx >= 1 else 0
    end = duration_boundaries[bin_idx]
    print(f"[{start}, {end}), weight: {w}")


if denoise_flag_check:
    label_configs.append(['noise_flag_old', 10001, 1, 1])
    label_configs.append(['noise_flag_new', 10000, 15, 15])


class DNNModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.autoint_layer = AutoIntModel(prefix='mhsar',
                                          attention_embedding_size=embedding_dim * len(attention_slot_bias),
                                          num_heads=1,
                                          num_blocks=1,
                                          dropout_rate=0.2)
        hadamard_slots = dict([(k, tuple(v)) for k, v in hadamard_slot.items()])
        self.hadamard_layer = HadamardLayer(hadamard_slots=hadamard_slots, task_names=task_names)
        self.bn_layer = tf.keras.layers.BatchNormalization(name="doc_input_layerbn")
        self.mmoe_model = MMoEModel(task_names=task_names,
                                    expert_num=4,
                                    expert_hidden_sizes=[512, 512, 256, 128],
                                    tower_hidden_sizes=[128, 1])
        self.denoise_net = {}
        for name in task_names:
            self.denoise_net[name] = DenoiseLayer(name=name)

        self.auto_group_layers = []
        for order, group_num in zip(ORDERS, GROUP_NUMS):
            self.auto_group_layers.append(
                AutoGroupLayer(slot_num=autogroup_slot_num,
                               group_num=group_num,
                               order=order,
                               temperature=temperature))

        # global step
        # self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False)

    def call(self, inputs, training):
        # dense features
        sparse_features = [inputs[feature_name] for feature_name in dnn_feature_names]
        # hadamard
        hadamard_deep_outs, hadamard_wide_outs = self.hadamard_layer(inputs)
        # autogroup
        autogroup_input = [inputs[feature_name] for feature_name in autogroup_feature_names] + hadamard_deep_outs
        autogroup_features = tf.stack(autogroup_input, axis=1)
        print("stack_feature shape:", autogroup_features.shape)  # (None, 202, 8)

        auto_group_outputs = []
        for auto_group_layer in self.auto_group_layers:
            # [batch_size, group_num, embedding_dim]
            auto_group_output = auto_group_layer(autogroup_features, training)
            shape = auto_group_output.get_shape().as_list()
            auto_group_outputs.append(tf.reshape(auto_group_output, [-1, shape[1] * shape[2]]))
        print(f"auto_group_outputs size: {len(auto_group_outputs)} shape:{auto_group_outputs[0].shape}")
        # attn
        attention_features = tf.concat([inputs['seq_{}'.format(seq_name)] for seq_name in seq_names], axis=2)
        attention_output = self.autoint_layer(attention_features, training)
        print("attention_output shape:", attention_output)  # (None, 1240)
        # concat
        embedding_merge_layer = tf.concat([attention_output] + sparse_features + hadamard_deep_outs
                                          + auto_group_outputs, axis=-1)
        embedding_merge_layer = self.bn_layer(embedding_merge_layer, training)
        print("embedding_merge_layer shape:", embedding_merge_layer.shape)  # (None, 3188)

        deep_outs = self.mmoe_model(embedding_merge_layer, training)
        wide_outs = [inputs[f"sparse_lr_{name}"] for name in task_names]
        print("concat shape: ", wide_outs[0].shape, hadamard_wide_outs[0].shape, deep_outs[0].shape)

        weights = []
        for biz, weight in biz_weights:
            weights.append(tf.where(inputs[f'biz_type_{biz}'] > 0, weight, 0.0))
        weights = tf.add_n(weights)
        weights = tf.where(tf.not_equal(weights, 0), weights, 1.0)

        duration_weights_tensor = tf.constant(duration_weights, dtype=tf.float32)
        bucket_indices = tf.raw_ops.Bucketize(input=inputs['duration'], boundaries=duration_boundaries)
        duration_weight = tf.nn.embedding_lookup(duration_weights_tensor, bucket_indices)     # 使用索引查找权重
        # 将权重应用到input_duration
        weights = tf.multiply(weights, duration_weight)
        # tf.print('duration weights:', tf.stack([inputs['debias_duration'], weights], axis=2), summarize=-1)

        ratio_out = tf.add_n([wide_outs[0], hadamard_wide_outs[0], deep_outs[0]])
        ratio_pred = tf.clip_by_value(tf.sigmoid(ratio_out) * weights, clip_value_min=0.0, clip_value_max=1.0)
        skip_out = tf.add_n([wide_outs[1], hadamard_wide_outs[1], deep_outs[1]])
        skip_pred = tf.clip_by_value(tf.sigmoid(skip_out) / weights, clip_value_min=0.0, clip_value_max=1.0)
        finish_out = tf.add_n([wide_outs[2], hadamard_wide_outs[2], deep_outs[2]])
        finish_pred = tf.clip_by_value(tf.sigmoid(finish_out) * weights, clip_value_min=0.0, clip_value_max=1.0)
        time_out = tf.add_n([wide_outs[3], hadamard_wide_outs[3], deep_outs[3]])
        time_pred = tf.clip_by_value(time_out * weights, clip_value_min=-10, clip_value_max=10)
        denoise_net_logits = {}
        for name in task_names:
            lr_input = inputs["weightnet_sparse_lr_{}".format(name)]
            denoise_net_logits[name] = self.denoise_net[name]((embedding_merge_layer, lr_input), training)

        out_sum = tf.concat([ratio_pred, finish_pred, skip_pred, time_pred], axis=1)
        predict_new = tf.concat([ratio_pred, finish_pred, skip_pred, time_pred], axis=1)
        predict = tf.concat([ratio_pred, finish_pred, skip_pred, time_pred], axis=1)

        return {'ratio_out': ratio_out,
                'skip_out': skip_out,
                'finish_out': finish_out,
                'time_out': time_out,
                'ratio_pred': ratio_pred,
                'skip_pred': skip_pred,
                'is_finish_pred': finish_pred,
                'time_pred': time_pred,
                'denoise_net_ratio_logit': denoise_net_logits['ratio'],
                'denoise_net_skip_logit': denoise_net_logits['skip'],
                'denoise_net_finish_logit': denoise_net_logits['finish'],
                'denoise_net_time_logit': denoise_net_logits['time'],
                'out_sum': out_sum,
                'predict_new': predict_new,
                'predict': predict
                }


class QBMiniFloatModel(NumerousModel):
    def __init__(self, name):
        super().__init__(name)
        self.losser = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.set_optimizer(tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, learning_rate=args.dnn_lr))
        self.dnn = DNNModel()
        self.wtg_updater = {}
        for task in wtg_tasks:
            self.wtg_updater[task] = WatchTimeGainUpdater(bin_boundaries=duration_boundaries, name=task)

        self.add_metric('auc_skip', numerous.metrics.AUC(name='auc_skip'))
        self.add_metric('believed_auc_skip', numerous.metrics.AUC(name='believed_auc_skip'))
        self.add_metric('auc_denoise_skip', numerous.metrics.AUC(name='auc_denoise_skip'))

        self.add_metric('skip_loss_mean', numerous.metrics.Mean(name='skip_loss_mean'))
        self.add_metric('denoise_net_skip_loss_mean', numerous.metrics.Mean(name='denoise_net_skip_loss_mean'))
        self.add_metric('final_skip_loss_mean', numerous.metrics.Mean(name='final_skip_loss_mean'))
        self.add_metric('final_believed_skip_loss_mean', numerous.metrics.Mean(name='final_believed_skip_loss_mean'))
        self.add_metric('skip_pred_mean', numerous.metrics.Mean(name='skip_pred_mean'))
        self.add_metric('believed_skip_pred_mean', numerous.metrics.Mean(name='believed_skip_pred_mean'))

        self.add_metric('auc_finishplay', numerous.metrics.AUC(name='auc_finishplay'))
        self.add_metric('believed_auc_is_finish', numerous.metrics.AUC(name='believed_auc_is_finish'))
        self.add_metric('auc_denoise_finish', numerous.metrics.AUC(name='auc_denoise_finish'))

        self.add_metric('finish_loss_mean', numerous.metrics.Mean(name='finish_loss_mean'))
        self.add_metric('denoise_net_finish_loss_mean', numerous.metrics.Mean(name='denoise_net_finish_loss_mean'))
        self.add_metric('final_is_finish_loss_mean', numerous.metrics.Mean(name='final_is_finish_loss_mean'))
        self.add_metric('final_believed_is_finish_loss_mean',
                        numerous.metrics.Mean(name='final_believed_is_finish_loss_mean'))
        self.add_metric('is_finish_pred_mean', numerous.metrics.Mean(name='is_finish_pred_mean'))
        self.add_metric('believed_is_finish_pred_mean', numerous.metrics.Mean(name='believed_is_finish_pred_mean'))

        self.add_metric('ratio_loss_mean', numerous.metrics.Mean(name='ratio_loss_mean'))
        self.add_metric('denoise_net_ratio_loss_mean', numerous.metrics.Mean(name='denoise_net_ratio_loss_mean'))
        self.add_metric('final_ratio_loss_mean', numerous.metrics.Mean(name='final_ratio_loss_mean'))
        self.add_metric('final_believed_ratio_loss_mean', numerous.metrics.Mean(name='final_believed_ratio_loss_mean'))
        self.add_metric('ratio_pred_mean', numerous.metrics.Mean(name='ratio_pred_mean'))
        self.add_metric('believed_ratio_pred_mean', numerous.metrics.Mean(name='believed_ratio_pred_mean'))

        self.add_metric('time_loss_mean', numerous.metrics.Mean(name='time_loss_mean'))
        self.add_metric('denoise_net_time_loss_mean', numerous.metrics.Mean(name='denoise_net_time_loss_mean'))
        self.add_metric('final_time_loss_mean', numerous.metrics.Mean(name='final_time_loss_mean'))
        self.add_metric('final_believed_time_loss_mean', numerous.metrics.Mean(name='final_believed_time_loss_mean'))
        self.add_metric('time_pred_mean', numerous.metrics.Mean(name='time_pred_mean'))
        self.add_metric('believed_time_pred_mean', numerous.metrics.Mean(name='believed_time_pred_mean'))

    def build_inputs(self):
        emb_initializer = numerous.initializers.RandomUniform(minval=-0.001, maxval=0.001)
        emb_optimizer = numerous.optimizers.Adam(beta_1=0.9, beta_2=0.999, learning_rate=args.emb_lr)
        ftrl_optimizer = numerous.optimizers.Ftrl(alpha=0.04, beta=1.0, lambda1=2.0, lambda2=1.0)
        inputs = Inputs(optimizer=emb_optimizer, initializer=emb_initializer)
        double_hashing_config = numerous.layers.DoubleHashingCompressConfig(hash_range=1 << 26)
        combiner = numerous.layers.Combiner.SUM

        old_feasign_feature_names = ["sparse_w_{}".format(slot_id) for slot_id in old_feasign_slots]
        inputs.add_share_embedding_features(
            feature_names=old_feasign_feature_names,
            slot_ids=old_feasign_slots,
            embedding_dim=embedding_dim,
            table_name="sparse_w",
            lengths=None,
            combiner=combiner,
            optimizer=emb_optimizer,
            initializer=emb_initializer,
            dump_dtype=numerous.int4
        )

        for slot_id in new_feasign_slots:
            inputs.add_sparse_feature(
                feature_name=f"sparse_w_{slot_id}",
                slot_id=slot_id,
                embedding_dim=embedding_dim,
                combiner=combiner,
                optimizer=emb_optimizer,
                initializer=emb_initializer,
                dump_dtype=numerous.int8
            )

        for slot_id in hash_slots:
            feature_name = f"sparse_w_{slot_id}"
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=embedding_dim,
                initializer=emb_initializer,
                table_name=feature_name,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.int4,
                double_hashing_config=double_hashing_config
            )

        for bias, target_slot, seq_name in zip(attention_slot_bias, target_slots, seq_names):
            seq_slots = [target_slot] + [slot + bias for slot in attention_item_slots]
            inputs.add_share_embedding_features(
                feature_names=[f"seq_{seq_name}_{slot}" for slot in seq_slots],
                slot_ids=seq_slots,
                embedding_dim=embedding_dim,
                table_name=f"seq_{seq_name}",
                lengths=None,
                combiner=combiner,
                optimizer=emb_optimizer,
                initializer=emb_initializer,
                dump_dtype=numerous.int4
            )

        for name in task_names:
            inputs.add_sparse_feature(
                feature_name="sparse_lr_{}".format(name),
                slot_id=lr_feature_slots,
                embedding_dim=1,
                initializer=emb_initializer,
                table_name="sparse_lr_{}".format(name),
                combiner=combiner,
                optimizer=ftrl_optimizer,
                zero_filter_threshold=1e-6,
                dump_dtype=numerous.float16
            )

        for name in task_names:
            inputs.add_sparse_feature(
                feature_name="weightnet_sparse_lr_{}".format(name),
                slot_id=lr_feature_slots,
                embedding_dim=1,
                initializer=emb_initializer,
                table_name="weightnet_sparse_lr_{}".format(name),
                combiner=combiner,
                optimizer=ftrl_optimizer,
                zero_filter_threshold=1e-6,
                dump_dtype=numerous.float16
            )

        for biz, w in biz_weights:
            inputs.add_dense_feature(feature_name=f'biz_type_{biz}',
                                     slot_id=biz_slot,
                                     lower_bound=biz, upper_bound=biz)

        for name, slot_id, lower_bound, upper_bound in duration_configs:
            inputs.add_dense_feature(feature_name=name,
                                     slot_id=slot_id,
                                     lower_bound=lower_bound, upper_bound=upper_bound)

        for task, slot_id, lower_bound, upper_bound in label_configs:
            inputs.add_label(label_name=task, slot_id=slot_id,
                             lower_bound=lower_bound, upper_bound=upper_bound)

        return inputs.build()

    def call(self, inputs, training):
        inputs = self.preprocess_inputs(inputs)  # process the features into a dictionary
        outputs = self.dnn(inputs, training)
        return outputs

    def compute_loss(self, labels, model_outputs, sample_weights, training):
        ratio_out = model_outputs['ratio_out']
        skip_out = model_outputs['skip_out']
        finish_out = model_outputs['finish_out']
        time_out = model_outputs['time_out']

        ratio_pred = tf.sigmoid(ratio_out)
        skip_pred = tf.sigmoid(skip_out)
        finish_pred = tf.sigmoid(finish_out)
        time_pred = time_out

        denoise_net_ratio_logit = model_outputs['denoise_net_ratio_logit']
        denoise_net_skip_logit = model_outputs['denoise_net_skip_logit']
        denoise_net_finish_logit = model_outputs['denoise_net_finish_logit']
        denoise_net_time_logit = model_outputs['denoise_net_time_logit']

        splaytime = tf.clip_by_value(labels["splaytime"], clip_value_min=0.0, clip_value_max=3600)
        stime = tf.clip_by_value(labels["stime"], clip_value_min=0.0, clip_value_max=3600)

        # ratio label
        ratio_label = tf.divide(splaytime, stime + 1e-8)
        ratio_label = tf.clip_by_value(ratio_label, 0.0, 1.0, name="ratio_label_clip")
        ratio_label = tf.cast(ratio_label, tf.float32, name="ratio_label")

        # time label
        # time_label = tf.clip_by_value(stime / 110.0, 0.0, 1.0, name="time_label_clip")
        # time_label = tf.cast(time_label, tf.float32, name="time_label")
        time_label = self.wtg_updater['time']([splaytime, stime], name='time_label')

        # skip label
        splaytime_shape = tf.fill(tf.shape(splaytime), 5.0)
        skip_label = tf.less(splaytime, splaytime_shape)
        skip_label = tf.cast(skip_label, tf.float32, name="skip_label")

        # finish label
        finish_label = tf.less_equal(stime, splaytime)
        finish_label = tf.cast(finish_label, tf.float32, name="finish_label")

        # gen noise sample flag
        action_sum = tf.add_n([labels["ispraise"], labels["isshare"], labels["isreadcomment"],
                               labels["iswritecomment"], labels["isfollow"], labels["isfeedback"]])
        noise_sample_flag = tf.logical_and(
            tf.logical_and(labels["stime"] == labels["splaytime"], labels['stime'] > 0), action_sum == 0)
        noise_sample_flag = tf.cast(noise_sample_flag, tf.float32, name="noise_sample_flag")
        weightnet_sample_mask = tf.reshape(noise_sample_flag < 0.5, [-1])  # noise sample: False, normal sample: True

        # ratio denoise_net loss + weight
        # tf.boolean_mask will filter the tensor by the mask=False
        denoise_net_ratio_label = tf.boolean_mask(ratio_label, weightnet_sample_mask)
        denoise_net_ratio_logit_mask = tf.boolean_mask(denoise_net_ratio_logit, weightnet_sample_mask)
        denoise_net_ratio_loss = tf.cond(tf.equal(tf.size(denoise_net_ratio_label), 0),
                                         lambda: tf.zeros_like(denoise_net_ratio_label),
                                         lambda: tf.nn.sigmoid_cross_entropy_with_logits(
                                             logits=denoise_net_ratio_logit_mask, labels=denoise_net_ratio_label))
        denoise_net_ratio_pred = tf.sigmoid(denoise_net_ratio_logit, "denoise_net_ratio_pred")
        denoise_net_ratio_weight = tf.reshape(
            tf.stop_gradient(denoise_net_ratio_pred, name="denoise_net_ratio_weight1"), [-1])
        denoise_net_ratio_weight = tf.where(weightnet_sample_mask, tf.ones_like(denoise_net_ratio_weight, tf.float32),
                                            denoise_net_ratio_weight, name="denoise_net_ratio_weight2")

        # skip denoise_net loss + weight
        denoise_net_skip_label = tf.boolean_mask(skip_label, weightnet_sample_mask)
        denoise_net_skip_masked_logit = tf.boolean_mask(denoise_net_skip_logit, weightnet_sample_mask)
        # denoise_net_skip_loss = tf.cond(tf.equal(tf.size(believed_skip_label), 0),
        #                                 lambda: tf.zeros_like(believed_skip_label),
        #                                 lambda: tf.nn.sigmoid_cross_entropy_with_logits(
        #                                     logits=denoise_net_skip_masked_logit,
        #                                     labels=believed_skip_label))
        denoise_net_skip_loss = tf.cond(tf.equal(tf.size(denoise_net_skip_label), 0),
                                        lambda: tf.zeros_like(denoise_net_skip_label),
                                        lambda: tf.math.square(denoise_net_skip_masked_logit - denoise_net_skip_label))
        denoise_net_skip_pred = tf.sigmoid(denoise_net_skip_logit, "denoise_net_skip_pred")
        denoise_net_skip_weight = tf.reshape(
            tf.stop_gradient(denoise_net_skip_pred, name="denoise_net_skip_weight1"), [-1])
        denoise_net_skip_weight = tf.where(weightnet_sample_mask, tf.zeros_like(denoise_net_skip_weight, tf.float32),
                                           denoise_net_skip_weight, name="denoise_net_skip_weight2")

        # finish denoise_net loss + weight
        denoise_net_finish_label = tf.boolean_mask(finish_label, weightnet_sample_mask)
        denoise_net_finish_logit_mask = tf.boolean_mask(denoise_net_finish_logit, weightnet_sample_mask)
        denoise_net_finish_loss = tf.cond(tf.equal(tf.size(denoise_net_finish_label), 0),
                                          lambda: tf.zeros_like(denoise_net_finish_label),
                                          lambda: tf.nn.sigmoid_cross_entropy_with_logits(
                                              logits=denoise_net_finish_logit_mask, labels=denoise_net_finish_label))
        denoise_net_finish_pred = tf.sigmoid(denoise_net_finish_logit, "denoise_net_finish_pred")
        denoise_net_finish_weight = tf.reshape(
            tf.stop_gradient(denoise_net_finish_pred, name="denoise_net_finish_weight1"), [-1])
        denoise_net_finish_weight = tf.where(weightnet_sample_mask, tf.ones_like(denoise_net_finish_weight, tf.float32),
                                             denoise_net_finish_weight, name="denoise_net_finish_weight2")

        # time denoise_net loss + weight
        denoise_net_time_label = tf.boolean_mask(time_label, weightnet_sample_mask)
        denoise_net_time_logit_mask = tf.boolean_mask(denoise_net_time_logit, weightnet_sample_mask)
        denoise_net_time_loss = tf.cond(tf.equal(tf.size(denoise_net_time_label), 0),
                                        lambda: tf.zeros_like(denoise_net_time_label),
                                        lambda: tf.math.square(denoise_net_time_logit_mask - denoise_net_time_label))
        denoise_net_time_pred = tf.sigmoid(denoise_net_time_logit, "denoise_net_time_pred")
        denoise_net_time_weight = tf.reshape(
            tf.stop_gradient(denoise_net_time_pred, name="denoise_net_time_weight1"), [-1])
        denoise_net_time_weight = tf.where(weightnet_sample_mask, tf.ones_like(denoise_net_time_weight, tf.float32),
                                           denoise_net_time_weight, name="denoise_net_time_weight2")

        # main network loss
        ratio_loss = tf.math.square(ratio_pred - ratio_label) * tf.reshape(denoise_net_ratio_weight, [-1, 1])
        skip_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=skip_out, labels=skip_label) * tf.reshape(1 - denoise_net_skip_weight, [-1, 1])
        finish_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=finish_out, labels=finish_label) * tf.reshape(denoise_net_finish_weight, [-1, 1])
        time_loss = tf.math.square(time_pred - time_label) * tf.reshape(denoise_net_time_weight, [-1, 1])

        final_loss = [ratio_loss, skip_loss, finish_loss, time_loss,
                      denoise_net_ratio_loss, denoise_net_skip_loss, denoise_net_finish_loss, denoise_net_time_loss]

        # skip metrics
        believed_skip_out = tf.boolean_mask(skip_out, weightnet_sample_mask)
        believed_skip_pred = tf.boolean_mask(skip_pred, weightnet_sample_mask)
        denoise_net_skip_masked_pred = tf.sigmoid(denoise_net_skip_masked_logit)
        final_skip_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=skip_out, labels=skip_label),
                                         name="final_skip_loss")
        final_believed_skip_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=believed_skip_out, labels=denoise_net_skip_label),
            name="final_believed_skip_loss")
        self.get_metric("auc_skip").update_state(skip_label, skip_pred)
        self.get_metric("believed_auc_skip").update_state(denoise_net_skip_label, believed_skip_pred)
        self.get_metric("auc_denoise_skip").update_state(denoise_net_skip_label, denoise_net_skip_masked_pred)
        self.get_metric('skip_loss_mean').update_state(skip_loss)
        self.get_metric('denoise_net_skip_loss_mean').update_state(denoise_net_skip_loss)
        self.get_metric('final_skip_loss_mean').update_state(final_skip_loss)  # auc_skip with no denoise weight
        self.get_metric('final_believed_skip_loss_mean').update_state(final_believed_skip_loss)
        self.get_metric('skip_pred_mean').update_state(skip_pred)
        self.get_metric('believed_skip_pred_mean').update_state(believed_skip_pred)

        # finish metrics
        believed_is_finish_out = tf.boolean_mask(finish_out, weightnet_sample_mask)
        believced_is_finish_pred = tf.boolean_mask(finish_pred, weightnet_sample_mask)
        believed_is_finish_label = denoise_net_finish_label
        denoise_net_is_finish_pred_mask = tf.sigmoid(denoise_net_finish_logit_mask)
        final_is_finish_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=finish_out, labels=finish_label),
            name="final_is_finish_loss")
        final_believed_is_finish_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=believed_is_finish_out, labels=believed_is_finish_label),
            name="final_believed_is_finish_loss")
        self.get_metric("auc_finishplay").update_state(finish_label, finish_pred)
        self.get_metric("believed_auc_is_finish").update_state(believed_is_finish_label, believced_is_finish_pred)
        self.get_metric("auc_denoise_finish").update_state(denoise_net_finish_label, denoise_net_is_finish_pred_mask)
        self.get_metric('finish_loss_mean').update_state(finish_loss)
        self.get_metric('denoise_net_finish_loss_mean').update_state(denoise_net_finish_loss)
        self.get_metric('final_is_finish_loss_mean').update_state(final_is_finish_loss)
        self.get_metric('final_believed_is_finish_loss_mean').update_state(final_believed_is_finish_loss)
        self.get_metric('is_finish_pred_mean').update_state(finish_pred)
        self.get_metric('believed_is_finish_pred_mean').update_state(believced_is_finish_pred)

        # ratio metrics
        believed_ratio_pred = tf.boolean_mask(ratio_pred, weightnet_sample_mask)
        believed_ratio_label = denoise_net_ratio_label
        final_ratio_loss = tf.losses.mean_squared_error(ratio_label, ratio_pred)
        final_believed_ratio_loss = tf.losses.mean_squared_error(believed_ratio_label, believed_ratio_pred)
        self.get_metric('ratio_loss_mean').update_state(ratio_loss)
        self.get_metric('denoise_net_ratio_loss_mean').update_state(denoise_net_ratio_loss)
        self.get_metric('final_ratio_loss_mean').update_state(final_ratio_loss)
        self.get_metric('final_believed_ratio_loss_mean').update_state(final_believed_ratio_loss)
        self.get_metric('ratio_pred_mean').update_state(ratio_pred)
        self.get_metric('believed_ratio_pred_mean').update_state(believed_ratio_pred)

        # time metrics
        believed_time_pred = tf.boolean_mask(time_pred, weightnet_sample_mask)
        believed_time_label = denoise_net_time_label
        final_time_loss = tf.losses.mean_squared_error(time_label, time_pred)
        final_believed_time_loss = tf.losses.mean_squared_error(believed_time_label, believed_time_pred)
        self.get_metric('time_loss_mean').update_state(time_loss)
        self.get_metric('denoise_net_time_loss_mean').update_state(denoise_net_time_loss)
        self.get_metric('final_time_loss_mean').update_state(final_time_loss)
        self.get_metric('final_believed_time_loss_mean').update_state(final_believed_time_loss)
        self.get_metric('time_pred_mean').update_state(time_pred)
        self.get_metric('believed_time_pred_mean').update_state(believed_time_pred)

        return final_loss


numerous.cluster.start(numerous.cluster.get_cluster_config(run_mode=numerous.cluster.RunMode.CLUSTER))
with numerous.distribute.get_strategy().scope():
    model = QBMiniFloatModel(name="qbminifloat")
    model.init_by_run_config(RunConfig(GPU_CONFIG))
    model.run()
numerous.cluster.stop()
