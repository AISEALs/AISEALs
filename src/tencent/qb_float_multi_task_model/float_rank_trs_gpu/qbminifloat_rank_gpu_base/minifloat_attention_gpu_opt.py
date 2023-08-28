#!/usr/bin/python3

import utils
from denoise_model import DenoiseLayer
from mmoe_model import MMoEModel
from autoint_model import AutoIntModel
from model_zoo.models import NumerousModel
from model_zoo.inputs import Inputs
import tensorflow.keras as K
import numerous
import tensorflow as tf
from numerous.train.run_config import RunConfig
from numerous.utils.config import get_config
import os

os.environ["NCCL_LAUNCH_MODE"] = "GROUP"


skip_recover_nn = int(get_config().get_option("userdefine.config.skip_recover_nn", default_value="0"))
skip_recover_nn = bool(skip_recover_nn)
do_recover_thread_num = int(get_config().get_option("userdefine.config.do_recover_thread_num", default_value="128"))

weight_factor = float(get_config().get_option("userdefine.config.weight_factor", default_value="1.0"))
print('weight factor is: ', weight_factor)

GPU_CARD_NUM = numerous.distribute.local_gpu_num()
GPU_CONFIG = {
    "is_auto_optimize": False,
    "reserved_gpu_memory_bytes": 1024 * 1024 * 1024 * 4,
    "max_gpu_memory_occupied_percentage": 0.8,
    "reader_config": {
        "one_ins_max_fea_num": 2048,
        "is_ignore_zero_key": False,
        "is_ignore_negative_factor_key": False,
        "is_ignore_non_batch_ins": False,
        "file_loader_thread_num": 24 * GPU_CARD_NUM,
        "extract_unique_key_thread_num": 16 * GPU_CARD_NUM,
        "merge_thread_num": 8 * GPU_CARD_NUM,
        "write_fileins_thread_num": 4 * GPU_CARD_NUM,
        "write_unique_key_thread_num": 4 * GPU_CARD_NUM,
        "sub_dir_file_regexs": [".*0", ".*1", ".*2", ".*3", ".*4", ".*5", ".*6", ".*7", ".*8", ".*9"],
        "pass_dir_regexs": [".*"],
        "dir_format": "%Y-%m-%d/%H",
        "data_format": "text",
        "sample_col_delimiter": "|",
        "sample_feature_delimiter": ";",
        "feature_section_delimiter": ":",
        "id_col_pos": 0,
        "label_col_pos": 6,
        "feature_col_pos": 7,
        "plugin_config": {},
    },
    "parameter_server_config": {
        "part_num": 1024,
        "hbm_max_key_num": 1024 * 1024 * 800,
        "hbm_max_dynamic_byte_size": 1024 * 1024 * 1024 * 28,
        "hbm_hashmap_load_factor": 0.75,
        "prepare_pass_thread_num": 12 * GPU_CARD_NUM,
        "build_pass_thread_num": 12 * GPU_CARD_NUM,
        "build_group_thread_num": 12 * GPU_CARD_NUM,
        "build_group_postprocess_thread_num": 4 * GPU_CARD_NUM,
        "do_persistence_thread_num": 24 * GPU_CARD_NUM,
        "load_checkpoint_thread_num": 16 * GPU_CARD_NUM,
        "use_parallel_optimizer": False,
        "delete_after_unseen_days": 7,
        "do_recover_thread_num": do_recover_thread_num,
        "cow_clip": {"r": 1.0, "lower_bound": 1e-5},
    },
    "avengers_manager_base_config": {
        "pipeline_batch_num": 2,
        "read_ins_thread_num": 4 * GPU_CARD_NUM,
        "convert_ins_thread_num": 8 * GPU_CARD_NUM,
        "compute_batch_thread_num": 1,
    },
    "saver_config": {
        "model_dump_interval": 1,
        # "dump_hotkey_size": 2 * (1 << 30),
        "dump_hotkey_size": 0,
        "always_complete": True,
        "streaming_model_dump_threads": 128,
        "is_streaming_model_dump": True,
        "skip_recover_nn": skip_recover_nn
    },
}

data_source = numerous.utils.config.get_config().get_option("data_source", default_value="hdfs")
if data_source == 'kafka' or data_source == 'cdmq':
    cdmq_interval_min = 6
    GPU_CONFIG['reader_config']['cdmq_consumer_thread_num'] = 1
    GPU_CONFIG['reader_config']['cdmq_interval_min'] = cdmq_interval_min
    GPU_CONFIG['reader_config']['cdmq_max_sample_num_per_pass'] = 500000 * cdmq_interval_min
    GPU_CONFIG['reader_config']['plugin_path'] = "libonline_sample_preprocessor.so"
    GPU_CONFIG['parameter_server_config']['is_serialized_cache'] = True
    GPU_CONFIG['saver_config']['model_dump_interval'] = 5
    GPU_CONFIG['saver_config']['always_complete'] = False
    GPU_CONFIG['saver_config']['complete_hour'] = '02,05,08,11,14,17,23'
    GPU_CONFIG['avengers_manager_base_config']['read_ins_thread_num'] = 12 * GPU_CARD_NUM
    GPU_CONFIG['avengers_manager_base_config']['convert_ins_thread_num'] = 16 * GPU_CARD_NUM

print('GPU_CONFIG:', GPU_CONFIG)

dnn_slots = [200, 201, 202, 203, 204, 205, 206, 208, 286, 553, 600, 2396, 616, 620, 621, 2397, 2401, 638, 639, 640, 641,
             646, 648, 2403, 2402, 697, 698, 699, 700, 707, 708, 712, 714, 715, 716, 717, 718, 722, 723, 724, 727, 728,
             729, 730, 731, 733, 738, 748, 750, 760, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711,
             1712, 1713, 1714, 1717, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732,
             1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751,
             1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769,
             1770, 2391, 1772, 1773, 1775, 1776, 2392, 2393, 1779, 1780, 1781, 1782, 1822, 1832, 1833, 1842, 1855, 1856,
             1857, 1858, 2404, 1860, 1861, 2398, 1863, 1864, 1865, 1866, 1867, 1868, 1869, 2399, 2400, 2395, 2394, 1874,
             1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910,
             1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1929, 1930, 1931, 1932,
             1933, 1934, 1935, 1936, 1941, 1942, 1943, 1944, 1945]
filter_slots = [205, 1726, 1932, 1933, 1934, 1945]
replaced_slot = 1942
dnn_feature_names = ['sparse_w_' + str(slot_id) if slot_id not in filter_slots else 'sparse_w_' + str(replaced_slot)
                     for slot_id in dnn_slots]

lr_slots = '619,621,2397,2401,638,639,646,648,2403,2402,699,700,707,708,712,714,715,728,729,730,738,744,745,746,748,750,1461,1691,1692,1693,1694,1695,1696,1697,1698,1699,1700,1701,1702,1703,1704,1707,1708,1709,1710,1711,1712,1713,1833,1842,2404,1860,1861,2398,1864,1865,1866,1868,1869,2399,2400,2395,2394,1874,1875,1876,1877,1878,1879,1901,1902,1903,1904,1905,1906,1907,1908,1909,1910,1911,1944'

# attention history slots
num_click_history = 30
item_start_slot = 1541
item_end_slot = item_start_slot + num_click_history - 1
attention_item_slots = utils.parseSlotsList(str(item_start_slot) + "-" + str(item_end_slot))
attention_slot_bias = [0, 90, 60, 30, 120]   # [item, pos, sub, tag, playtime]
target_slots = [600, 625, 697, 1832, 624]
# [target_item, target_pos, target_sub, target_tag, target_playtime, target_vtime]
seq_names = ['item', 'pos', 'sub', 'tag', 'playtime']

double_hash_slots = [1866, 1867]

embedding_dim = 8
rate = 0.2
num_tasks = 3
task_names = ['ratio', 'skip', 'finish']
layersizes = [512, 512, 256, 128]
num_experts = 4

label_configs = [['video_time', 10000, 1, 1],
                 ['play_time', 10000, 2, 2],
                 ['ratio', 10000, 3, 3],
                 ['ispraise', 10000, 9, 9],
                 ['isshare', 10000, 10, 10],
                 ['isreadcomment', 10000, 11, 11],
                 ['iswritecomment', 10000, 12, 12],
                 ['isfollow', 10000, 13, 13],
                 ['isfeedback', 10000, 14, 14],
                 ]

novel_keys = [2670426682886, 2669348800384, 2670821175892, 2670755561568, 2668453849280, 2669438389676, 2670097486576,
              2667441101032, 2667495773662]
novel_slot = 621

long_video_keys = [1]
long_video_slot = 1947


class QBMiniFLoatModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.autoint_model = AutoIntModel(prefix='mhsar',
                                          attention_embedding_size=embedding_dim * len(attention_slot_bias),
                                          num_heads=1,
                                          num_blocks=1,
                                          dropout_rate=rate)
        self.bn_layer = tf.keras.layers.BatchNormalization(name="doc_input_layerbn")
        self.mmoe_model = MMoEModel(layersizes=layersizes, num_experts=num_experts, task_names=task_names,
                                    layer_name='doc')
        self.towers = []
        for i, task_name in enumerate(task_names):
            self.towers.append(
                K.models.Sequential([
                    K.layers.Dense(128, activation=None, name="top_layer1_{}".format(i)),
                    K.layers.BatchNormalization(name="layer1_bn{}".format(i)),
                    K.layers.Activation('relu'),
                    K.layers.Dense(1, activation='relu', name="top_layer2_{}".format(i)),
                ], name='tower_{}'.format(i)))

        self.denoise_net = {}
        for name in task_names:
            self.denoise_net[name] = DenoiseLayer(layer_name=name)

    def get_attention_layer(self, inputs):
        attention_slot_embedding_dict = {}
        for slot_id in attention_item_slots:
            attention_embeddings = []
            for bias in range(0, len(attention_slot_bias)):
                cur_slot_id = slot_id + attention_slot_bias[bias]
                attention_embedding = inputs["sparse_{}_attention_w_{}".format(bias, cur_slot_id)]
                attention_embeddings.append(attention_embedding)
            attention_slot_embedding_dict[slot_id] = tf.concat(
                attention_embeddings, axis=1)
        target_embeddings = []
        for slot_index in range(0, len(target_slots)):
            target_embedding = inputs["sparse_{}_attention_w_{}".format(slot_index, target_slots[slot_index])]
            target_embeddings.append(target_embedding)
        attention_embedding_size = embedding_dim * len(target_slots)
        attention_slot_embedding_dict[target_slots[0]] = tf.concat(
            target_embeddings, axis=1)
        attention_slot_embedding_concat_layer = tf.concat(
            list(attention_slot_embedding_dict.values()), axis=1)
        embedding_attention_layer = tf.reshape(attention_slot_embedding_concat_layer, shape=[tf.shape(
            attention_slot_embedding_concat_layer)[0], len(attention_item_slots) + 1, attention_embedding_size])
        return embedding_attention_layer

    def call(self, inputs, training):
        dense_embedding_layer = tf.concat([inputs[feature_name] for feature_name in dnn_feature_names], axis=1,
                                          name='dense_embedding')
        attention_embedding_layer = self.get_attention_layer(inputs)
        attention_output = self.autoint_model(attention_embedding_layer, training)
        embedding_merge_layer = tf.concat([attention_output, dense_embedding_layer], axis=1)
        embedding_merge_layer_bn = self.bn_layer(embedding_merge_layer, training)
        tower_inputs = self.mmoe_model(embedding_merge_layer_bn, training)
        deep_outs = []
        for i, task_name in enumerate(task_names):
            deep_outs.append(self.towers[i](tower_inputs[i]))

        wide_outs = [inputs["sparse_lr_{}".format(index)] for index in range(1, num_tasks + 1)]

        novel_flag = []
        for i in range(len(novel_keys)):
            novel_flag.append(inputs['novel_flag_{}'.format(i)])
        novel_flag = tf.greater(tf.add_n(novel_flag), 0.0)

        long_video_flag = []
        for i in range(len(long_video_keys)):
            long_video_flag.append(inputs['long_video_flag_{}'.format(i)])
        long_video_flag = tf.greater(tf.add_n(long_video_flag), 0.0)

        reweight_flag = tf.logical_or(novel_flag, long_video_flag)

        ratio_out = tf.add(wide_outs[0], deep_outs[0])
        ratio_pred = tf.sigmoid(ratio_out)
        ratio_pred = tf.where(reweight_flag,
                              tf.clip_by_value(ratio_pred * weight_factor, clip_value_min=0.0, clip_value_max=1.0),
                              ratio_pred)

        skip_out = tf.add(wide_outs[1], deep_outs[1])
        skip_pred = tf.sigmoid(skip_out)

        skip_pred = tf.where(reweight_flag,
                             tf.clip_by_value(skip_pred / weight_factor, clip_value_min=0.0, clip_value_max=1.0),
                             skip_pred)

        finish_out = tf.add(wide_outs[2], deep_outs[2])
        finish_pred = tf.sigmoid(finish_out)
        finish_pred = tf.where(reweight_flag,
                               tf.clip_by_value(finish_pred * weight_factor, clip_value_min=0.0, clip_value_max=1.0),
                               finish_pred)

        denoise_net_logits = {}
        for name in task_names:
            lr_input = inputs["weightnet_sparse_lr_{}".format(name)]
            denoise_net_logits[name] = self.denoise_net[name]((embedding_merge_layer_bn, lr_input), training)

        out_sum = tf.concat([ratio_pred, finish_pred, skip_pred], axis=1)
        predict_new = tf.concat([ratio_pred, finish_pred, skip_pred], axis=1)
        predict = tf.concat([ratio_pred, finish_pred, skip_pred], axis=1)

        return {'ratio_out': ratio_out,
                'skip_out': skip_out,
                'finish_out': finish_out,
                'ratio_pred': ratio_pred,
                'skip_pred': skip_pred,
                'isfinish_pred': finish_pred,
                'is_finish_pred': finish_pred,
                'denoise_net_ratio_logit': denoise_net_logits['ratio'],
                'denoise_net_skip_logit': denoise_net_logits['skip'],
                'denoise_net_finish_logit': denoise_net_logits['finish'],
                'out_sum': out_sum,
                'predict_new': predict_new,
                'predict': predict
                }


class DNNModel(NumerousModel):
    def __init__(self, name):
        super().__init__(name)
        self.losser = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.set_optimizer(tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, learning_rate=0.0001))
        self.dnn = QBMiniFLoatModel()
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

    def build_inputs(self):
        emb_initializer = numerous.initializers.RandomUniform(minval=-0.001, maxval=0.001)
        emb_optimizer = numerous.optimizers.Adam(beta_1=0.9, beta_2=0.999, learning_rate=0.001)
        ftrl_optimizer = numerous.optimizers.Ftrl(alpha=0.04, beta=1.0, lambda1=2.0, lambda2=1.0)
        inputs = Inputs(optimizer=emb_optimizer, initializer=emb_initializer)

        combiner = numerous.layers.Combiner.SUM

        for slot_id in dnn_slots:
            feature_name = "sparse_w_{}".format(slot_id)
            double_hashing_config = None
            if slot_id in double_hash_slots:
                double_hashing_config = numerous.layers.DoubleHashingCompressConfig(hash_range=1 << 26)
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=str(slot_id),
                embedding_dim=embedding_dim,
                initializer=emb_initializer,
                table_name=feature_name,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.int8,
                double_hashing_config=double_hashing_config
            )

        for slot_id in attention_item_slots:
            for bias in range(0, len(attention_slot_bias)):
                cur_slot_id = slot_id + attention_slot_bias[bias]
                feature_name = "sparse_{}_attention_w_{}".format(bias, cur_slot_id)
                inputs.add_sparse_feature(
                    feature_name=feature_name,
                    slot_id=str(cur_slot_id),
                    embedding_dim=embedding_dim,
                    initializer=emb_initializer,
                    table_name=feature_name,
                    combiner=combiner,
                    optimizer=emb_optimizer,
                    dump_dtype=numerous.int8
                )
        for slot_index in range(0, len(target_slots)):
            feature_name = "sparse_{}_attention_w_{}".format(slot_index, target_slots[slot_index])
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=str(target_slots[slot_index]),
                embedding_dim=embedding_dim,
                initializer=emb_initializer,
                table_name=feature_name,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.int8
            )

        for index, task_name in enumerate(task_names):
            inputs.add_sparse_feature(
                feature_name="sparse_lr_{0}".format(index + 1),
                slot_id=lr_slots,
                embedding_dim=1,
                initializer=emb_initializer,
                table_name="sparse_lr_{0}".format(index + 1),
                combiner=combiner,
                optimizer=ftrl_optimizer,
                zero_filter_threshold=1e-6
            )

        for name in task_names:
            inputs.add_sparse_feature(
                feature_name="weightnet_sparse_lr_{}".format(name),
                slot_id=lr_slots,
                embedding_dim=1,
                initializer=emb_initializer,
                table_name="weightnet_sparse_lr_{}".format(name),
                combiner=combiner,
                optimizer=ftrl_optimizer,
                zero_filter_threshold=1e-6
            )

        for task, slot_id, lower_bound, upper_bound in label_configs:
            inputs.add_label(label_name=task, slot_id=slot_id,
                             lower_bound=lower_bound, upper_bound=upper_bound)

        for i, key in enumerate(novel_keys):
            inputs.add_dense_feature(feature_name='novel_flag_{}'.format(i),
                                     slot_id=novel_slot,
                                     lower_bound=key, upper_bound=key)
        for i, key in enumerate(long_video_keys):
            inputs.add_dense_feature(feature_name='long_video_flag_{}'.format(i),
                                     slot_id=long_video_slot,
                                     lower_bound=key, upper_bound=key)

        return inputs.build()

    def call(self, inputs, training):
        inputs = self.preprocess_inputs(inputs)  # process the features into a dictionary
        outputs = self.dnn(inputs, training)
        return outputs

    def compute_loss(self, labels, model_outputs, sample_weights, training):
        ratio_out = model_outputs['ratio_out']
        skip_out = model_outputs['skip_out']
        finish_out = model_outputs['finish_out']

        ratio_pred = tf.sigmoid(ratio_out)
        skip_pred = tf.sigmoid(skip_out)
        finish_pred = tf.sigmoid(finish_out)

        denoise_net_ratio_logit = model_outputs['denoise_net_ratio_logit']
        denoise_net_skip_logit = model_outputs['denoise_net_skip_logit']
        denoise_net_finish_logit = model_outputs['denoise_net_finish_logit']

        # noise_label = labels['noise_label']
        play_time = labels['play_time']
        video_time = labels['video_time']
        # ratio = labels['ratio']

        # labels
        # ratio label
        ratio_label = play_time / (video_time + 1e-8)
        ratio_label = tf.clip_by_value(ratio_label, 0.0, 1.0, name="ratio_label_clip")
        ratio_label = tf.cast(ratio_label, tf.float32, name="ratio_label")

        # skip label
        skip_label = tf.less(play_time, tf.fill(tf.shape(play_time), 5.0))
        skip_label = tf.cast(skip_label, tf.float32, name="skip_label")

        # finish label
        finish_label = tf.less_equal(video_time, play_time)
        finish_label = tf.cast(finish_label, tf.float32, name="finish_label")

        action_sum = tf.add_n([labels["ispraise"], labels["isshare"], labels["isreadcomment"],
                               labels["iswritecomment"], labels["isfollow"], labels["isfeedback"]])
        noise_sample_flag = tf.logical_and(
            tf.logical_and(labels["play_time"] == labels["video_time"], labels['video_time'] > 0), action_sum == 0)
        noise_sample_flag = tf.cast(noise_sample_flag, tf.float32, name="noise_sample_flag")
        weightnet_sample_mask = tf.reshape(noise_sample_flag < 0.5, [-1])

        denoise_net_ratio_label = ratio_label
        denoise_net_ratio_label = tf.boolean_mask(
            denoise_net_ratio_label, weightnet_sample_mask
        )
        denoise_net_ratio_logit_mask = tf.boolean_mask(
            denoise_net_ratio_logit, weightnet_sample_mask
        )

        denoise_net_ratio_loss = tf.cond(
            tf.equal(tf.size(denoise_net_ratio_label), 0),
            lambda: tf.zeros_like(denoise_net_ratio_label),
            lambda: tf.nn.sigmoid_cross_entropy_with_logits(
                logits=denoise_net_ratio_logit_mask, labels=denoise_net_ratio_label
            ),
        )
        denoise_net_ratio_pred = tf.sigmoid(
            denoise_net_ratio_logit, "denoise_net_ratio_pred"
        )
        denoise_net_ratio_weight = tf.reshape(tf.stop_gradient(
            denoise_net_ratio_pred, name="denoise_net_ratio_weight1"
        ), [-1])
        denoise_net_ratio_weight = tf.where(
            weightnet_sample_mask,
            tf.ones_like(denoise_net_ratio_weight, tf.float32),
            denoise_net_ratio_weight,
            name="denoise_net_ratio_weight2",
        )

        # skip denoise net
        denoise_net_skip_label = skip_label
        denoise_net_skip_label = tf.boolean_mask(
            denoise_net_skip_label, weightnet_sample_mask
        )
        denoise_net_skip_logit_mask = tf.boolean_mask(
            denoise_net_skip_logit, weightnet_sample_mask
        )
        denoise_net_skip_pred_mask = tf.sigmoid(denoise_net_skip_logit_mask)

        denoise_net_skip_loss = tf.cond(
            tf.equal(tf.size(denoise_net_skip_label), 0),
            lambda: tf.zeros_like(denoise_net_skip_label),
            lambda: tf.nn.sigmoid_cross_entropy_with_logits(
                logits=denoise_net_skip_logit_mask, labels=denoise_net_skip_label
            ),
        )
        denoise_net_skip_pred = tf.sigmoid(denoise_net_skip_logit, "denoise_net_skip_pred")
        denoise_net_skip_weight = tf.reshape(tf.stop_gradient(
            denoise_net_skip_pred, name="denoise_net_skip_weight1"
        ), [-1])
        denoise_net_skip_weight = tf.where(
            weightnet_sample_mask,
            tf.zeros_like(denoise_net_skip_weight, tf.float32),
            denoise_net_skip_weight,
            name="denoise_net_skip_weight2",
        )

        # finish denoise net
        denoise_net_finish_label = finish_label
        denoise_net_finish_label = tf.boolean_mask(
            denoise_net_finish_label, weightnet_sample_mask
        )
        denoise_net_finish_logit_mask = tf.boolean_mask(
            denoise_net_finish_logit, weightnet_sample_mask
        )
        denoise_net_finish_pred_mask = tf.sigmoid(denoise_net_finish_logit_mask)

        denoise_net_finish_loss = tf.cond(
            tf.equal(tf.size(denoise_net_finish_label), 0),
            lambda: tf.zeros_like(denoise_net_finish_label),
            lambda: tf.nn.sigmoid_cross_entropy_with_logits(
                logits=denoise_net_finish_logit_mask, labels=denoise_net_finish_label
            ),
        )
        denoise_net_finish_pred = tf.sigmoid(
            denoise_net_finish_logit, "denoise_net_finish_pred"
        )
        denoise_net_finish_weight = tf.reshape(tf.stop_gradient(
            denoise_net_finish_pred, name="denoise_net_finish_weight1"
        ), [-1])
        denoise_net_finish_weight = tf.where(
            weightnet_sample_mask,
            tf.ones_like(denoise_net_finish_weight, tf.float32),
            denoise_net_finish_weight,
            name="denoise_net_finish_weight2"
        )

        ratio_loss = tf.math.square(ratio_pred - ratio_label) * tf.reshape(denoise_net_ratio_weight, [-1, 1])
        skip_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=skip_out, labels=skip_label) * tf.reshape(1 - denoise_net_skip_weight, [-1, 1])
        finish_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=finish_out, labels=finish_label) * tf.reshape(denoise_net_finish_weight, [-1, 1])

        final_loss = [ratio_loss, skip_loss, finish_loss, denoise_net_ratio_loss, denoise_net_skip_loss,
                      denoise_net_finish_loss]

        # skip metrics
        believed_skip_out = tf.boolean_mask(skip_out, weightnet_sample_mask)
        believed_skip_pred = tf.boolean_mask(skip_pred, weightnet_sample_mask)
        believed_skip_label = denoise_net_skip_label
        denoise_net_skip_pred_mask = tf.sigmoid(denoise_net_skip_logit_mask)
        final_skip_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=skip_out, labels=skip_label),
                                         name="final_skip_loss")
        final_believed_skip_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=believed_skip_out, labels=believed_skip_label),
            name="final_believed_skip_loss")

        self.get_metric("auc_skip").update_state(skip_label, skip_pred)
        self.get_metric("believed_auc_skip").update_state(believed_skip_label, believed_skip_pred)
        self.get_metric("auc_denoise_skip").update_state(denoise_net_skip_label, denoise_net_skip_pred_mask)

        self.get_metric('skip_loss_mean').update_state(skip_loss)
        self.get_metric('denoise_net_skip_loss_mean').update_state(denoise_net_skip_loss)
        self.get_metric('final_skip_loss_mean').update_state(final_skip_loss)
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
            tf.nn.sigmoid_cross_entropy_with_logits(logits=believed_is_finish_out, labels=believed_skip_label),
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
        return final_loss


numerous.cluster.start(numerous.cluster.get_cluster_config(run_mode=numerous.cluster.RunMode.CLUSTER))
with numerous.distribute.get_strategy().scope():
    model = DNNModel(name="qbminifloat")
    model.init_by_run_config(RunConfig(GPU_CONFIG))
    model.run()
numerous.cluster.stop()
