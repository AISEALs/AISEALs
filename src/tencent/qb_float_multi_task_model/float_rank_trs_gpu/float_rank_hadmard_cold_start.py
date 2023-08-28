#!/usr/bin/python3
import utils
from layers.denoise_model import DenoiseLayer
from layers.mmoe_model import MMoEModel
from layers.autoint_model import AutoIntModel
from model_zoo.models import NumerousModel
from model_zoo.inputs import Inputs
import tensorflow.keras as K
import numerous
import tensorflow as tf
from numerous.layers.embedding import GPUEmbeddingOptions, CowClipOption
from numerous.train.run_config import set_run_config, RunConfig, GPUOptions
from numerous.data.reader_config import ReaderConfig, GPUReaderOptions, set_reader_config, PreprocessPlugin
from numerous.train.saver import DumpType, Saver
from numerous.utils.config import get_config
import os

# GPU setting
GPU_CARD_NUM = numerous.distribute.local_gpu_num()
os.environ["NCCL_LAUNCH_MODE"] = "GROUP"

is_auto_optimize = bool(int(get_config().get_option("userdefine.config.is_auto_optimize", default_value="0")))
skip_recover_nn = bool(int(get_config().get_option("userdefine.config.skip_recover_nn", default_value="0")))
do_recover_thread_num = int(get_config().get_option("userdefine.config.do_recover_thread_num", default_value="128"))
data_format = get_config().get_option("userdefine.config.data_format", default_value="row")
pass_num = int(get_config().get_option("userdefine.config.pass_num", default_value="10"))
if pass_num == 10:
    sub_dir_file_regexes = [".*part-{:01d}-.*".format(i) for i in range(10)]
else:
    sub_dir_file_regexes = [".*part-{:01d}-.*".format(i) for i in range(5)]
data_source = get_config().get_option("data_source", default_value="hdfs")
cdmq_hdfs_base_path = get_config().get_option("userdefine.config.cdmq_hdfs_base_path",
                                              default_value="mdfs://cloudhdfs/mttsparknew/data/video/zhihuawang/rank/sample")
kafka_consumer_thread_num = int(get_config().get_option("userdefine.config.kafka_consumer_thread_num", default_value="8"))
kafka_pass_interval_min = int(get_config().get_option("userdefine.config.kafka_pass_interval_min", default_value="6"))
kafka_max_pass_example_num = int(get_config().get_option("userdefine.config.kafka_max_pass_example_num",
                                                         default_value=str(500000 * 6)))
embedding_persistence_thread_num = int(get_config().get_option("userdefine.config.embedding_persistence_thread_num",
                                                               default_value=str(24 * GPU_CARD_NUM)))
embedding_key_partition_num = int(get_config().get_option("userdefine.config.embedding_key_partition_num", default_value="256"))
max_reserved_gpu_memory_fraction = float(get_config().get_option("userdefine.config.max_reserved_gpu_memory_fraction",
                                                                 default_value="0.2"))
max_reserved_cpu_memory_fraction = float(get_config().get_option("userdefine.config.max_reserved_cpu_memory_fraction",
                                                                 default_value="0.2"))

# slots
dnn_slots = [
    200, 201, 202, 203, 204, 205, 206, 208, 286, 553, 600, 2396, 616, 619, 620, 621, 2397, 2401, 638, 639, 640, 641, 646, 648,
    2403, 2402, 697, 698, 699, 700, 707, 708, 712, 714, 715, 716, 717, 718, 722, 723, 724, 727, 728, 729, 730, 731, 733, 738,
    748, 750, 760, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1717, 1719, 1720, 1721,
    1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742,
    1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762,
    1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 2391, 1772, 1773, 1775, 1776, 2392, 2393, 1779, 1780, 1781, 1782, 1822,
    1832, 1833, 1842, 1855, 1856, 1857, 1858, 2404, 1860, 1861, 2398, 1863, 1864, 1865, 1866, 1867, 1868, 1869, 2399, 2400,
    2395, 2394, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909,
    1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1929, 1930, 1931, 1932, 1933,
    1934, 1935, 1936, 1941, 1942, 1943, 1944, 1945, 1949
]
# filter_slots 为失效 slot, 使用 1942 来替代。
filter_slots = [205, 1726, 1932, 1933, 1934, 1945]
replaced_slot = 1942
dnn_feature_names = []
for slot_id in dnn_slots:
    if slot_id not in filter_slots and slot_id != 1949:
        dnn_feature_names.append('sparse_w_' + str(slot_id))
    elif slot_id != 1949:
        dnn_feature_names.append('sparse_w_' + str(replaced_slot))
lr_slots = '619,621,2397,2401,638,639,646,648,2403,2402,699,700,707,708,712,714,715,728,729,730,738,744,745,746,748,750,1461,1691,1692,1693,1694,1695,1696,1697,1698,1699,1700,1701,1702,1703,1704,1707,1708,1709,1710,1711,1712,1713,1833,1842,2404,1860,1861,2398,1864,1865,1866,1868,1869,2399,2400,2395,2394,1874,1875,1876,1877,1878,1879,1901,1902,1903,1904,1905,1906,1907,1908,1909,1910,1911,1944'
# attention history slots
num_click_history = 30
item_start_slot = 1541
item_end_slot = item_start_slot + num_click_history - 1
attention_item_slots = utils.parseSlotsList(str(item_start_slot) + "-" + str(item_end_slot))
attention_slot_bias = [0, 90, 60, 30, 120]  # [item, pos, sub, tag, playtime]
target_slots = [600, 625, 697, 1832, 624]
seq_names = ['item', 'pos', 'sub', 'tag', 'playtime']
double_hash_slots = [1866, 1867]
hadamard_slot = {1879: [1832, 1909], 640: [600, 204], 1866: [1856, 600], 1867: [1856, 1832]}
novel_slot = 621
long_video_keys = [1]
long_video_slot = 1947

# model setting
embedding_dim = 8
rate = 0.2
num_tasks = 3
task_names = ['ratio', 'skip', 'finish']
layersizes = [512, 512, 256, 128]
num_experts = 4
label_configs = [
    ["video_time", 10000, 1, 1],
    ["play_time", 10000, 2, 2],
    ["ispraise", 10000, 9, 9],
    ["isshare", 10000, 10, 10],
    ["isreadcomment", 10000, 11, 11],
    ["iswritecomment", 10000, 12, 12],
    ["isfollow", 10000, 13, 13],
    ["isfeedback", 10000, 14, 14],
]
novel_keys = [
    2670426682886, 2669348800384, 2670821175892, 2670755561568, 2668453849280, 2669438389676, 2670097486576, 2667441101032,
    2667495773662
]


class QBMiniFLoatModel(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.autoint_model = AutoIntModel(prefix='mhsar',
                                          attention_embedding_size=embedding_dim * len(attention_slot_bias),
                                          num_heads=1,
                                          num_blocks=1,
                                          dropout_rate=rate)
        self.autoint_model_for_item_id_embedding = AutoIntModel(prefix='mhsar',
                                                                attention_embedding_size=embedding_dim,
                                                                num_heads=1,
                                                                num_blocks=1,
                                                                dropout_rate=rate)
        self.autoint_model_for_user_id_embedding = AutoIntModel(prefix='mhsar',
                                                                attention_embedding_size=embedding_dim,
                                                                num_heads=1,
                                                                num_blocks=1,
                                                                dropout_rate=rate)
        self.item_generalize_embedding_layer = tf.keras.layers.Dense(units=embedding_dim, activation='relu')
        self.user_generalize_embedding_layer = tf.keras.layers.Dense(units=embedding_dim, activation='relu')
        self.bn_layer = tf.keras.layers.BatchNormalization(name="doc_input_layerbn")
        self.mmoe_model = MMoEModel(layersizes=layersizes, num_experts=num_experts, task_names=task_names, layer_name='doc')
        self.towers = []
        for i, task_name in enumerate(task_names):
            self.towers.append(
                K.models.Sequential([
                    K.layers.Dense(128, activation=None, name="top_layer1_{}".format(i)),
                    K.layers.BatchNormalization(name="layer1_bn{}".format(i)),
                    K.layers.Activation('relu'),
                    K.layers.Dense(1, activation='relu', name="top_layer2_{}".format(i)),
                ],
                                    name='tower_{}'.format(i)))
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
            attention_slot_embedding_dict[slot_id] = tf.concat(attention_embeddings, axis=1)
        target_embeddings = []
        for slot_index in range(0, len(target_slots)):
            target_embedding = inputs["sparse_{}_attention_w_{}".format(slot_index, target_slots[slot_index])]
            target_embeddings.append(target_embedding)
        attention_embedding_size = embedding_dim * len(target_slots)
        attention_slot_embedding_dict[target_slots[0]] = tf.concat(target_embeddings, axis=1)
        attention_slot_embedding_concat_layer = tf.concat(list(attention_slot_embedding_dict.values()), axis=1)
        embedding_attention_layer = tf.reshape(
            attention_slot_embedding_concat_layer,
            shape=[tf.shape(attention_slot_embedding_concat_layer)[0],
                   len(attention_item_slots) + 1, attention_embedding_size])
        return embedding_attention_layer

    def get_cold_start_id_embedding(self, inputs, training):
        item_feature_names = [
            "sparse_w_616",
            "sparse_w_619",
            "sparse_w_620",
            "sparse_w_697",
            "sparse_w_698",
            "sparse_w_731",
            "sparse_w_733",
            "sparse_w_1702",
            "sparse_w_1703",
            "sparse_w_1832",
        ]
        item_generalization_embedding = self.item_generalize_embedding_layer(
            tf.concat([inputs[feature_name] for feature_name in item_feature_names], axis=-1))
        item_keys = tf.stack(
            [inputs["sparse_w_600"], inputs["sparse_w_1949"], item_generalization_embedding],
            axis=1,
        )  # (bsz, feature_num, hidden_size)
        cold_start_item_embedding = self.autoint_model_for_item_id_embedding(item_keys, training)

        user_feature_names = [
            "sparse_w_201", "sparse_w_202", "sparse_w_203", "sparse_w_204", "sparse_w_205", "sparse_w_206", "sparse_w_208",
            "sparse_w_553", "sparse_w_1855", "sparse_w_1856", "sparse_w_1857", "sparse_w_1858", "sparse_w_1931",
            "sparse_w_1932", "sparse_w_1933", "sparse_w_1934", "sparse_w_1935", "sparse_w_1936", "sparse_w_1941",
            "sparse_w_1942", "sparse_w_1943", "sparse_w_1945"
        ]
        user_generalization_embedding = self.user_generalize_embedding_layer(
            tf.concat([inputs[feature_name] for feature_name in user_feature_names], axis=-1))
        user_clk_history = tf.reduce_mean(
            tf.stack(
                [inputs["sparse_0_attention_w_{}".format(item_slot)] for item_slot in attention_item_slots],
                axis=1,
            ),
            axis=1,
        )  # (bsz, hidden_size)
        user_keys = tf.stack(
            [inputs["sparse_w_200"], user_clk_history, user_generalization_embedding],
            axis=1,
        )  # (bsz, feature_num, hidden_size)
        cold_start_user_embedding = self.autoint_model_for_user_id_embedding(user_keys, training)

        return (cold_start_item_embedding, cold_start_user_embedding)

    def call(self, inputs, training):
        dense_embedding_layer = tf.concat([
            inputs[feature_name]
            for feature_name in dnn_feature_names if feature_name != "sparse_w_200" and feature_name != "sparse_w_600"
        ],
                                          axis=1,
                                          name='dense_embedding')

        attention_embedding_layer = self.get_attention_layer(inputs)
        attention_output = self.autoint_model(attention_embedding_layer, training)

        cold_start_item_embedding, cold_start_user_embedding = self.get_cold_start_id_embedding(inputs, training)

        embedding_merge_layer = tf.concat(
            [attention_output, dense_embedding_layer, cold_start_item_embedding, cold_start_user_embedding], axis=1)
        embedding_merge_layer_bn = self.bn_layer(embedding_merge_layer, training)
        tower_inputs = self.mmoe_model(embedding_merge_layer_bn, training)
        deep_outs = []
        for i, task_name in enumerate(task_names):
            deep_outs.append(self.towers[i](tower_inputs[i]))
        wide_outs = [inputs["sparse_lr_{}".format(index)] for index in range(1, num_tasks + 1)]
        ratio_out = tf.add(wide_outs[0], deep_outs[0])
        ratio_pred = tf.sigmoid(ratio_out)
        skip_out = tf.add(wide_outs[1], deep_outs[1])
        skip_pred = tf.sigmoid(skip_out)
        finish_out = tf.add(wide_outs[2], deep_outs[2])
        finish_pred = tf.sigmoid(finish_out)
        denoise_net_logits = {}
        for name in task_names:
            lr_input = inputs["weightnet_sparse_lr_{}".format(name)]
            denoise_net_logits[name] = self.denoise_net[name]((embedding_merge_layer_bn, lr_input), training)
        out_sum = tf.concat([ratio_pred, finish_pred, skip_pred], axis=1)
        predict_new = tf.concat([ratio_pred, finish_pred, skip_pred], axis=1)
        predict = tf.concat([ratio_pred, finish_pred, skip_pred], axis=1)
        return {
            'ratio_out': ratio_out,
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
        self.set_optimizer(tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, learning_rate=0.00005))
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
        self.add_metric('final_believed_is_finish_loss_mean', numerous.metrics.Mean(name='final_believed_is_finish_loss_mean'))
        self.add_metric('is_finish_pred_mean', numerous.metrics.Mean(name='is_finish_pred_mean'))
        self.add_metric('believed_is_finish_pred_mean', numerous.metrics.Mean(name='believed_is_finish_pred_mean'))
        self.add_metric('ratio_loss_mean', numerous.metrics.Mean(name='ratio_loss_mean'))
        self.add_metric('denoise_net_ratio_loss_mean', numerous.metrics.Mean(name='denoise_net_ratio_loss_mean'))
        self.add_metric('final_ratio_loss_mean', numerous.metrics.Mean(name='final_ratio_loss_mean'))
        self.add_metric('final_believed_ratio_loss_mean', numerous.metrics.Mean(name='final_believed_ratio_loss_mean'))
        self.add_metric('ratio_pred_mean', numerous.metrics.Mean(name='ratio_pred_mean'))
        self.add_metric('believed_ratio_pred_mean', numerous.metrics.Mean(name='believed_ratio_pred_mean'))
        if data_source == 'kafka' or data_source == 'cdmq':
            self.set_saver(
                numerous.train.Saver(
                    recover_config=numerous.train.RecoverConfig(recover_thread_num=do_recover_thread_num,
                                                                skip_recover_nn=skip_recover_nn),
                    dumper_config=numerous.train.DumperConfig(dump_interval=5,
                                                              dump_hotkey_size=0,
                                                              dump_type=DumpType.MIXED,
                                                              async_dump_threads=128,
                                                              dump_full_at_hour='02,05,08,11,14,17,23'),
                    expire_config=numerous.train.FeatureExpireConfig(expire_time=7 * 24 * 60 * 60),
                ))
        else:
            self.set_saver(
                numerous.train.Saver(
                    recover_config=numerous.train.RecoverConfig(recover_thread_num=do_recover_thread_num,
                                                                skip_recover_nn=skip_recover_nn),
                    dumper_config=numerous.train.DumperConfig(dump_interval=5,
                                                              dump_hotkey_size=0,
                                                              dump_type=DumpType.FULL,
                                                              async_dump_threads=128),
                    expire_config=numerous.train.FeatureExpireConfig(expire_time=7 * 24 * 60 * 60),
                ))

    def build_inputs(self):
        emb_initializer = numerous.initializers.RandomUniform(minval=-0.001, maxval=0.001)
        emb_optimizer = numerous.optimizers.Adam(beta_1=0.9, beta_2=0.999, learning_rate=0.001, use_parallel_version=False)
        ftrl_optimizer = numerous.optimizers.Ftrl(alpha=0.04, beta=1.0, lambda1=2.0, lambda2=1.0, use_parallel_version=False)
        if data_source == 'kafka' or data_source == 'cdmq':
            inputs = Inputs(optimizer=emb_optimizer,
                            initializer=emb_initializer,
                            gpu_embedding_options=GPUEmbeddingOptions(
                                embedding_key_partition_num=embedding_key_partition_num,
                                max_gpu_memory_key_num=1024 * 1024 * 800,
                                max_gpu_memory_key_bytes=1024 * 1024 * 1024 * 25,
                                gpu_memory_hashmap_load_factor=0.75,
                                prepare_pass_thread_num=12 * GPU_CARD_NUM,
                                build_pass_thread_num=12 * GPU_CARD_NUM,
                                build_group_thread_num=24 * GPU_CARD_NUM,
                                build_group_postprocess_thread_num=4 * GPU_CARD_NUM,
                                embedding_persistence_thread_num=embedding_persistence_thread_num,
                                grad_clip_option=CowClipOption(r_coeff=1.0, lower_bound=1e-5),
                                is_serialized_embedding_cache=True))
        else:
            inputs = Inputs(optimizer=emb_optimizer,
                            initializer=emb_initializer,
                            gpu_embedding_options=GPUEmbeddingOptions(
                                embedding_key_partition_num=embedding_key_partition_num,
                                max_gpu_memory_key_num=1024 * 1024 * 800,
                                max_gpu_memory_key_bytes=1024 * 1024 * 1024 * 25,
                                gpu_memory_hashmap_load_factor=0.75,
                                prepare_pass_thread_num=12 * GPU_CARD_NUM,
                                build_pass_thread_num=12 * GPU_CARD_NUM,
                                build_group_thread_num=24 * GPU_CARD_NUM,
                                build_group_postprocess_thread_num=4 * GPU_CARD_NUM,
                                embedding_persistence_thread_num=embedding_persistence_thread_num,
                                grad_clip_option=CowClipOption(r_coeff=1.0, lower_bound=1e-5),
                            ))
        combiner = numerous.layers.Combiner.SUM
        for slot_id in dnn_slots:
            feature_name = "sparse_w_{}".format(slot_id)
            double_hashing_config = None
            if slot_id in double_hash_slots:
                double_hashing_config = numerous.layers.DoubleHashingCompressConfig(hash_range=15000000)
            if slot_id in hadamard_slot:
                continue
            inputs.add_sparse_feature(feature_name=feature_name,
                                      slot_id=str(slot_id),
                                      embedding_dim=embedding_dim,
                                      initializer=emb_initializer,
                                      table_name=feature_name,
                                      combiner=combiner,
                                      optimizer=emb_optimizer,
                                      dump_dtype=numerous.int8,
                                      double_hashing_config=double_hashing_config)
        for slot_id in attention_item_slots:
            for bias in range(0, len(attention_slot_bias)):
                cur_slot_id = slot_id + attention_slot_bias[bias]
                feature_name = "sparse_{}_attention_w_{}".format(bias, cur_slot_id)
                inputs.add_sparse_feature(feature_name=feature_name,
                                          slot_id=str(cur_slot_id),
                                          embedding_dim=embedding_dim,
                                          initializer=emb_initializer,
                                          table_name=feature_name,
                                          combiner=combiner,
                                          optimizer=emb_optimizer,
                                          dump_dtype=numerous.int8)
        for slot_index in range(0, len(target_slots)):
            feature_name = "sparse_{}_attention_w_{}".format(slot_index, target_slots[slot_index])
            inputs.add_sparse_feature(feature_name=feature_name,
                                      slot_id=str(target_slots[slot_index]),
                                      embedding_dim=embedding_dim,
                                      initializer=emb_initializer,
                                      table_name=feature_name,
                                      combiner=combiner,
                                      optimizer=emb_optimizer,
                                      dump_dtype=numerous.int8)
        for index, task_name in enumerate(task_names):
            inputs.add_sparse_feature(feature_name="sparse_lr_{0}".format(index + 1),
                                      slot_id=lr_slots,
                                      embedding_dim=1,
                                      initializer=emb_initializer,
                                      table_name="sparse_lr_{0}".format(index + 1),
                                      combiner=combiner,
                                      optimizer=ftrl_optimizer,
                                      zero_filter_threshold=1e-6)
        for name in task_names:
            inputs.add_sparse_feature(feature_name="weightnet_sparse_lr_{}".format(name),
                                      slot_id=lr_slots,
                                      embedding_dim=1,
                                      initializer=emb_initializer,
                                      table_name="weightnet_sparse_lr_{}".format(name),
                                      combiner=combiner,
                                      optimizer=ftrl_optimizer,
                                      zero_filter_threshold=1e-6)
        for task, slot_id, lower_bound, upper_bound in label_configs:
            inputs.add_label(label_name=task, slot_id=slot_id, lower_bound=lower_bound, upper_bound=upper_bound)
        return inputs.build()

    def call(self, inputs, training):
        # process the features into a dictionary
        inputs = self.preprocess_inputs(inputs)
        for slot_id in hadamard_slot:
            inputs['sparse_w_' +
                   str(slot_id)] = inputs['sparse_w_' + str(hadamard_slot[slot_id][0])] * inputs['sparse_w_' +
                                                                                                 str(hadamard_slot[slot_id][1])]
            print('hadamard_slot ', inputs['sparse_w_' + str(slot_id)].get_shape(),
                  inputs['sparse_w_' + str(hadamard_slot[slot_id][0])].get_shape())
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

        play_time = labels['play_time']
        video_time = labels['video_time']

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

        # noise label
        action_sum = tf.add_n([
            labels["ispraise"],
            labels["isshare"],
            labels["isreadcomment"],
            labels["iswritecomment"],
            labels["isfollow"],
            labels["isfeedback"],
        ])
        noise_sample_flag = tf.logical_and(
            tf.logical_and(video_time == play_time, video_time > 0),
            action_sum == 0,
        )
        noise_sample_flag = tf.cast(noise_sample_flag, tf.float32, name="noise_sample_flag")

        weightnet_sample_mask = tf.reshape(noise_sample_flag < 0.5, [-1])  # 置信样本标志

        denoise_net_ratio_label = ratio_label
        denoise_net_ratio_label = tf.boolean_mask(denoise_net_ratio_label, weightnet_sample_mask)
        denoise_net_ratio_logit_mask = tf.boolean_mask(denoise_net_ratio_logit, weightnet_sample_mask)

        denoise_net_ratio_loss = tf.cond(
            tf.equal(tf.size(denoise_net_ratio_label), 0),
            lambda: tf.zeros_like(denoise_net_ratio_label),
            lambda: tf.nn.sigmoid_cross_entropy_with_logits(logits=denoise_net_ratio_logit_mask, labels=denoise_net_ratio_label
                                                            ),
        )
        denoise_net_ratio_pred = tf.sigmoid(denoise_net_ratio_logit, "denoise_net_ratio_pred")
        denoise_net_ratio_weight = tf.reshape(tf.stop_gradient(denoise_net_ratio_pred, name="denoise_net_ratio_weight1"), [-1])
        denoise_net_ratio_weight = tf.where(
            weightnet_sample_mask,
            tf.ones_like(denoise_net_ratio_weight, tf.float32),
            denoise_net_ratio_weight,
            name="denoise_net_ratio_weight2",
        )

        # skip denoise net
        denoise_net_skip_label = skip_label
        denoise_net_skip_label = tf.boolean_mask(denoise_net_skip_label, weightnet_sample_mask)
        denoise_net_skip_logit_mask = tf.boolean_mask(denoise_net_skip_logit, weightnet_sample_mask)
        denoise_net_skip_pred_mask = tf.sigmoid(denoise_net_skip_logit_mask)

        denoise_net_skip_loss = tf.cond(
            tf.equal(tf.size(denoise_net_skip_label), 0),
            lambda: tf.zeros_like(denoise_net_skip_label),
            lambda: tf.nn.sigmoid_cross_entropy_with_logits(logits=denoise_net_skip_logit_mask, labels=denoise_net_skip_label),
        )
        denoise_net_skip_pred = tf.sigmoid(denoise_net_skip_logit, "denoise_net_skip_pred")
        denoise_net_skip_weight = tf.reshape(tf.stop_gradient(denoise_net_skip_pred, name="denoise_net_skip_weight1"), [-1])
        denoise_net_skip_weight = tf.where(
            weightnet_sample_mask,
            tf.zeros_like(denoise_net_skip_weight, tf.float32),
            denoise_net_skip_weight,
            name="denoise_net_skip_weight2",
        )

        # finish denoise net
        denoise_net_finish_label = finish_label
        denoise_net_finish_label = tf.boolean_mask(denoise_net_finish_label, weightnet_sample_mask)
        denoise_net_finish_logit_mask = tf.boolean_mask(denoise_net_finish_logit, weightnet_sample_mask)
        # denoise_net_finish_pred_mask = tf.sigmoid(denoise_net_finish_logit_mask)

        denoise_net_finish_loss = tf.cond(
            tf.equal(tf.size(denoise_net_finish_label), 0),
            lambda: tf.zeros_like(denoise_net_finish_label),
            lambda: tf.nn.sigmoid_cross_entropy_with_logits(logits=denoise_net_finish_logit_mask,
                                                            labels=denoise_net_finish_label),
        )
        denoise_net_finish_pred = tf.sigmoid(denoise_net_finish_logit, "denoise_net_finish_pred")
        denoise_net_finish_weight = tf.reshape(tf.stop_gradient(denoise_net_finish_pred, name="denoise_net_finish_weight1"),
                                               [-1])
        denoise_net_finish_weight = tf.where(weightnet_sample_mask,
                                             tf.ones_like(denoise_net_finish_weight, tf.float32),
                                             denoise_net_finish_weight,
                                             name="denoise_net_finish_weight2")

        ratio_loss = tf.math.square(ratio_pred - ratio_label) * tf.reshape(denoise_net_ratio_weight, [-1, 1])
        skip_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=skip_out, labels=skip_label) * tf.reshape(
            1 - denoise_net_skip_weight, [-1, 1])
        finish_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=finish_out, labels=finish_label) * tf.reshape(
            denoise_net_finish_weight, [-1, 1])

        final_loss = [
            ratio_loss, skip_loss, finish_loss, denoise_net_ratio_loss, denoise_net_skip_loss, denoise_net_finish_loss
        ]

        # skip metrics
        believed_skip_out = tf.boolean_mask(skip_out, weightnet_sample_mask)
        believed_skip_pred = tf.boolean_mask(skip_pred, weightnet_sample_mask)
        believed_skip_label = denoise_net_skip_label
        denoise_net_skip_pred_mask = tf.sigmoid(denoise_net_skip_logit_mask)
        final_skip_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=skip_out, labels=skip_label),
                                         name="final_skip_loss")
        final_believed_skip_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=believed_skip_out,
                                                                                          labels=believed_skip_label),
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
        final_is_finish_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=finish_out, labels=finish_label),
                                              name="final_is_finish_loss")
        final_believed_is_finish_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=believed_is_finish_out,
                                                                                               labels=believed_skip_label),
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


if is_auto_optimize:
    run_config = numerous.train.RunConfig(
        gpu_options=GPUOptions(max_nn_gpu_memory_bytes=1024 * 1024 * 1024 * 4,
                               max_reserved_gpu_memory_fraction=max_reserved_gpu_memory_fraction,
                               max_reserved_cpu_memory_fraction=max_reserved_cpu_memory_fraction,
                               enable_auto_optimize=is_auto_optimize,
                               cdmq_hdfs_base_path=cdmq_hdfs_base_path))
else:
    run_config = numerous.train.RunConfig(gpu_options=GPUOptions(
        max_nn_gpu_memory_bytes=1024 * 1024 * 1024 * 4, max_reserved_gpu_memory_fraction=max_reserved_gpu_memory_fraction))
set_run_config(run_config)
if data_source == 'kafka' or data_source == 'cdmq':
    reader_config = numerous.data.ReaderConfig(gpu_options=GPUReaderOptions(
        max_example_feature_num=2048,
        ignore_zero_key=False,
        ignore_negative_key=False,
        ignore_example_less_one_batch=False,
        pass_dir_regexes=[".*"],
        sub_file_regexes_in_pass=sub_dir_file_regexes,
        group_dir_regexes=[".*"],
        dir_format="%Y-%m-%d/%H",
        data_format='text' if data_format == 'row' else 'parquet',
        sample_col_delimiter="|",
        sample_feature_delimiter=";",
        feature_section_delimiter=":",
        id_col_pos=0,
        label_col_pos=6,
        feature_col_pos=7,
        preprocess_plugin=PreprocessPlugin(plugin_config={}, plugin_path="libonline_sample_preprocessor.so"),
        pipeline_batch_num=2,
        read_example_thread_num=8 * GPU_CARD_NUM,
        convert_example_thread_num=12 * GPU_CARD_NUM,
        compute_batch_thread_num=1,
        extract_unique_key_thread_num=8 * GPU_CARD_NUM,
        merge_unique_key_thread_num=4 * GPU_CARD_NUM,
        write_example_file_thread_num=2 * GPU_CARD_NUM,
        write_unique_key_thread_num=8 * GPU_CARD_NUM,
        parse_file_chan_capacity=4 * GPU_CARD_NUM,
        example_loader_thread_num=8 * GPU_CARD_NUM,
        kafka_consumer_thread_num=kafka_consumer_thread_num,
        kafka_pass_interval_min=kafka_pass_interval_min,
        kafka_max_pass_example_num=kafka_max_pass_example_num))
else:
    reader_config = numerous.data.ReaderConfig(gpu_options=GPUReaderOptions(
        max_example_feature_num=2048,
        ignore_zero_key=False,
        ignore_negative_key=False,
        ignore_example_less_one_batch=False,
        pass_dir_regexes=[".*"],
        sub_file_regexes_in_pass=sub_dir_file_regexes,
        group_dir_regexes=[".*"],
        dir_format="%Y-%m-%d/%H",
        data_format='text' if data_format == 'row' else 'parquet',
        sample_col_delimiter="|",
        sample_feature_delimiter=";",
        feature_section_delimite=":",
        id_col_po=0,
        label_col_pos=6,
        feature_col_pos=7,
        preprocess_plugin=PreprocessPlugin(plugin_config={}),
        pipeline_batch_num=2,
        read_example_thread_num=4 * GPU_CARD_NUM,
        convert_example_thread_num=8 * GPU_CARD_NUM,
        compute_batch_thread_num=1,
        extract_unique_key_thread_num=8 * GPU_CARD_NUM,
        merge_unique_key_thread_num=4 * GPU_CARD_NUM,
        write_example_file_thread_num=2 * GPU_CARD_NUM,
        write_unique_key_thread_num=8 * GPU_CARD_NUM,
        parse_file_chan_capacity=4 * GPU_CARD_NUM,
        example_loader_thread_num=8 * GPU_CARD_NUM,
    ))
set_reader_config(reader_config)
numerous.cluster.start(numerous.cluster.get_cluster_config(run_mode=numerous.cluster.RunMode.CLUSTER))
with numerous.distribute.get_strategy().scope():
    model = DNNModel(name="qbminifloat")
    # model.init_by_run_config(RunConfig(GPU_CONFIG))
    model.run()
numerous.cluster.stop()
