#!/usr/bin/python3

import utils
from config.config import get_common_args
from layers import MMoEModel, DenoiseLayer, ShowClickLayer, AutoIntModel
from model_zoo.models import NumerousModel
from model_zoo.inputs import Inputs
import numerous
import tensorflow as tf
from numerous.train.run_config import RunConfig
from numerous.utils.config import get_config
import os

os.environ["NCCL_LAUNCH_MODE"] = "GROUP"

skip_recover_nn = int(get_config().get_option("userdefine.config.skip_recover_nn", default_value="0"))
skip_recover_nn = bool(skip_recover_nn)
do_recover_thread_num = int(get_config().get_option("userdefine.config.do_recover_thread_num", default_value="128"))
data_format = get_config().get_option("userdefine.config.data_format", default_value="row")
pass_num = int(get_config().get_option("userdefine.config.pass_num", default_value="10"))
card_ratio = int(get_config().get_option("userdefine.config.card_ratio", default_value="2"))
open_hotkey = int(get_config().get_option("userdefine.config.open_hotkey", default_value="0"))
print_interval = int(get_config().get_option("userdefine.config.print_interval", "0"))
showclick_check = int(get_config().get_option("userdefine.config.showclick_check", default_value="0"))
showclick_check = bool(showclick_check)
showclick_norm_type = get_config().get_option("userdefine.config.showclick_norm_type", default_value=None)
showclick_drop_rate = float(get_config().get_option("userdefine.config.showclick_drop_rate", default_value="0.0"))
showclick_use_share = int(get_config().get_option("userdefine.config.showclick_use_share", default_value="0"))
showclick_use_share = bool(showclick_use_share)
showclick_decay_rate = float(get_config().get_option("userdefine.config.showclick_decay_rate", default_value="0.999"))
showclick_decay_weight = float(get_config().get_option("userdefine.config.showclick_decay_weight",
                                                       default_value="0.0001"))
showclick_filter_topK = int(get_config().get_option("userdefine.config.showclick_filter_topK",
                                                    default_value="1"))
biz_debias = get_config().get_option("userdefine.config.biz_debias", default_value="3:1.2_4,5:1.2")
debug_mode = int(get_config().get_option("userdefine.config.debug_mode", default_value="0"))
debug_mode = bool(debug_mode)
file_loader_thread_num = int(get_config().get_option("userdefine.config.file_loader_thread_num", default_value="4"))
parse_file_chan_capacity = int(get_config().get_option("userdefine.config.parse_file_chan_capacity", default_value="5"))

print("skip_recover_nn: ", skip_recover_nn)
print("do_recover_thread_num: ", do_recover_thread_num)
print("data_format: ", data_format)
print("pass_num: ", pass_num)
print("card_ratio: ", card_ratio)
print("open_hotkey:", open_hotkey)
print("print_interval: ", print_interval)
print("showclick_check: ", showclick_check)
print("showclick_norm_type: ", showclick_norm_type)
print("showclick_drop_rate: ", showclick_drop_rate)
print("showclick_use_share: ", showclick_use_share)
print("showclick_decay_rate: ", showclick_decay_rate)
print('biz_debias: ', biz_debias)
print("showclick_decay_weight:", showclick_decay_weight)
print("showclick_filter_topK:", showclick_filter_topK)
print("debug_mode: ", debug_mode)

# https://iwiki.woa.com/pages/viewpage.action?pageId=1524230667
# sub_dir_file_regexes = ['part-.*{:01d}.parquet'.format(i) for i in range(10)]
if pass_num == 10:
    sub_dir_file_regexes = ['.*part-.*{:01d}(.parquet)?'.format(i) for i in range(10)]
else:
    sub_dir_file_regexes = ['.*part-{:01d}-.*'.format(i) for i in range(5)]
group_dir_regexs = [sub_dir_file_regexes[0]]
# group_dir_regexs = [sub_dir_file_regexes[0], sub_dir_file_regexes[3]]
# group_dir_regexs = ['.*']

args = get_common_args()
GPU_CARD_NUM = numerous.distribute.local_gpu_num()
GPU_CONFIG = {
    "is_auto_optimize": False,
    "reserved_gpu_memory_bytes": 1024 * 1024 * 1024 * 4,
    "max_gpu_memory_occupied_percentage": 0.8,
    "reader_config": {
        # 是否忽略样本中值为0的key（特征），由于有些数据集在生成样本时会使用0做为一些默认的key的值，可能需要在模型中剔除掉，
        # 此时可以设置这个为True。
        "is_ignore_zero_key": False,
        # 是否忽略样本中权重为负数的key（特征），由于有些数据集生成的过程中有些异常数据，使用-1表示，所以需要在模型中剔除掉，
        # 此时可以设置这个为True。
        "is_ignore_negative_factor_key": False,
        # 是否忽略一个pass（目录）中不够一个batch的剩余的样本，该参数一般用在含有batch norm的模型中，
        # 由于样本没法完全凑到minibatch的整数倍，而最后的一个不够的batch对于batch norm的参数影响比较大
        # 此时用户可以选择忽略掉这个batch，由于一般一个pass总的数据要远远大于minibatch，所以忽略掉这个数据对效果基本没有什么影响。
        "is_ignore_non_batch_ins": True,
        "one_ins_max_fea_num": 2048,
        # 用户读取样本的线程数，该配置在配置时需要考虑两个因素，如果开大的话那么读取的线程数变多，读取数据速度会快，同时会带来内存增加的风险，
        # 如果我们的样本的每个文件都比较大，而且训练过程中出现了OOM，那么建议将该值减小。
        "file_loader_thread_num": file_loader_thread_num,
        # 从用户的样本中抽取unique的key的线程数,默认的8 * GPU卡数
        "extract_unique_key_thread_num": card_ratio * 16 * GPU_CARD_NUM,
        # 从多个unique key的线程中对抽取出的key进行merge,默认的4 * GPU卡数
        "merge_thread_num": card_ratio * 8 * GPU_CARD_NUM,
        # 将转化后的样本写入到ssd中的线程数, 默认的2 * GPU卡数
        "write_fileins_thread_num": card_ratio * 16 * GPU_CARD_NUM,
        # 将抽取的unique的key写入到ssd中的线程数，默认的2 * GPU卡数
        "write_unique_key_thread_num": card_ratio * 8 * GPU_CARD_NUM,
        # 解析样本时使用的管道大小
        "parse_file_chan_capacity": parse_file_chan_capacity,
        "sub_dir_file_regexs": sub_dir_file_regexes,
        "group_dir_regexs": group_dir_regexs,
        "pass_dir_regexs": [".*"],
        "dir_format": "%Y-%m-%d/%H",
        "data_format": 'text' if data_format == 'row' else 'parquet',
        "sample_col_delimiter": "|",
        "sample_feature_delimiter": ";",
        "feature_section_delimiter": ":",
        "id_col_pos": 0,
        "label_col_pos": 6,
        "feature_col_pos": 7,
        "plugin_config": {},
    },
    "parameter_server_config": {
        "part_num": 2048,
        "hbm_max_key_num": 1024 * 1024 * 400 * GPU_CARD_NUM,  # Pass FeaKeyCount
        "hbm_max_dynamic_byte_size": 1024 * 1024 * 1024 * 12 * GPU_CARD_NUM,  # DynamicByteSize
        "hbm_hashmap_load_factor": 0.75,
        "prepare_pass_thread_num": card_ratio * 12 * GPU_CARD_NUM,  # 预准备下一个pass的参数的时候的线程数
        "build_pass_thread_num": card_ratio * 12 * GPU_CARD_NUM,  # 构建一个pass的参数的时候的线程数
        "build_group_thread_num": card_ratio * 12 * GPU_CARD_NUM,  # 构建一个group的参数的时候的线程数
        "build_group_postprocess_thread_num": card_ratio * 4 * GPU_CARD_NUM,  # 处理完一个group后的一些处理过程的线程数
        "do_persistence_thread_num": 24 * GPU_CARD_NUM,  # 对sparse的参数做持久化的线程数
        "load_checkpoint_thread_num": 16 * GPU_CARD_NUM,  # 加载sparse参数checkpoint的线程数
        "use_parallel_optimizer": False,
        "delete_after_unseen_days": 7,  # 2
        "do_recover_thread_num": do_recover_thread_num,
        "is_use_parallel_optimizer": False,  # 要使用GlobalSum优化器，这个配置必须为False.
        "cow_clip": {"r": 1.0, "lower_bound": 1e-5},
    },
    "avengers_manager_base_config": {
        "pipeline_batch_num": 2,  # 流水线中缓存的batch个数
        "read_ins_thread_num": card_ratio * 4 * GPU_CARD_NUM,  # 训练过程中用于从ssd中按照batch读取样本的线程数
        "convert_ins_thread_num": card_ratio * 4 * GPU_CARD_NUM,  # 训练过程中将读取的原始的batch样本转化为gpu所需要的数据格式的线程数
        "compute_batch_thread_num": 1,  # 训练过程中gpu用于执行训练的线程数
    },
    "saver_config": {
        "model_dump_interval": 5,  # hdfs模式下是多少个目录导出一次模型，cdmq模式下是多少个pass导出一次模型。
        "dump_hotkey_size": 2 * (1 << 30) if open_hotkey else 0,
        "always_complete": True,
        "streaming_model_dump_threads": 128,
        "is_streaming_model_dump": True,
        "skip_recover_nn": skip_recover_nn
    },
}

data_source = numerous.utils.config.get_config().get_option("data_source", default_value="hdfs")
if data_source == 'kafka' or data_source == 'cdmq':
    cdmq_interval_min = 6
    GPU_CONFIG['reader_config']['cdmq_consumer_thread_num'] = 8
    GPU_CONFIG['reader_config']['cdmq_interval_min'] = cdmq_interval_min
    GPU_CONFIG['reader_config']['cdmq_max_sample_num_per_pass'] = 500000 * cdmq_interval_min
    # GPU_CONFIG['reader_config']['plugin_path'] = "libonline_sample_preprocessor.so"
    GPU_CONFIG['saver_config']['model_dump_interval'] = 5
    GPU_CONFIG['saver_config']['always_complete'] = False
    GPU_CONFIG['saver_config']['complete_hour'] = '02,05,08,11,14,17,20,23'

print("GPU_CONFIG:", GPU_CONFIG)

dnn_slots = [200, 201, 202, 203, 204, 206, 208, 286, 553, 600, 2396, 616, 620, 621, 2397, 2401, 638, 639, 641,
             646, 648, 2403, 2402, 697, 698, 699, 700, 707, 708, 712, 714, 715, 716, 717, 718, 722, 723, 724, 727, 728,
             729, 730, 731, 733, 738, 748, 750, 760, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711,
             1712, 1713, 1714, 1717, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1727, 1728, 1729, 1730, 1731, 1732,
             1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751,
             1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769,
             1770, 2391, 1772, 1773, 1775, 1776, 2392, 2393, 1780, 1781, 1782, 1822, 1832, 1833, 1842, 1855, 1856,
             1857, 1858, 2404, 1860, 1861, 2398, 1863, 1864, 1865, 1868, 1869, 2399, 2400, 2395, 2394, 1874,
             1875, 1876, 1877, 1878, 1880, 1881, 1882, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910,
             1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1929, 1930, 1931, 1935,
             1936, 1941, 1942, 1943, 1944]

new_feasign_slots = [1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961]
new_feasign_feature_names = ['sparse_w_' + str(slot_id) for slot_id in new_feasign_slots]

lr_slots = '619,621,2397,2401,638,639,646,648,2403,2402,699,700,707,708,712,714,715,728,729,730,738,744,745,746,748,' \
           '750,1461,1691,1692,1693,1694,1695,1696,1697,1698,1699,1700,1701,1702,1703,1704,1707,1708,1709,1710,1711,' \
           '1712,1713,1833,1842,2404,1860,1861,2398,1864,1865,1868,1869,2399,2400,2395,2394,1874,1875,1876,1877,' \
           '1878,1901,1902,1903,1904,1905,1906,1907,1908,1909,1910,1911,1944'

# attention history slots
num_click_history = 30
item_start_slot = 1541
item_end_slot = item_start_slot + num_click_history - 1
attention_item_slots = utils.parseSlotsList(str(item_start_slot) + "-" + str(item_end_slot))
attention_slot_bias = [0, 90, 60, 30, 120]  # [item, pos, sub, tag, playtime]
target_slots = [600, 625, 697, 1832,
                624]  # [target_item, target_pos, target_sub, target_tag, target_playtime, target_vtime]
seq_names = ['item', 'pos', 'sub', 'tag', 'playtime']

hash_slots = [1779]  # [1866, 1867]
hadamard_slot = {1879: [1832, 1858],
                 640: [600, 204],
                 1866: [1856, 600],
                 1867: [1856, 1832],
                 4001: [1960, 1956],  # 4001-4008: 兴趣点交叉
                 4002: [1960, 1957],  # 1960: USER_INTEREST_STRONG_VEC_FSS
                 4003: [1960, 1958],  # 1961: USER_INTEREST_WEAK_VEC_FSS
                 4004: [1960, 1959],  # 1956: ITEM_INTEREST_FIRST_FSS
                 4005: [1961, 1956],  # 1957: ITEM_INTEREST_SECOND_FSS
                 4006: [1961, 1957],  # 1958: ITEM_INTEREST_THIRD_FSS
                 4007: [1961, 1958],  # 1959: ITEM_INTEREST_VEC_FSS
                 4008: [1961, 1959],
                 }
embedding_dim = 8
num_tasks = 3
task_names = ['ratio', 'skip', 'finish']

replaced_slot = 1942
# show_click_filter_slots = [712]   # 为了能够继承训练
show_click_filter_slots = [712, 1880, 1882, 1780, 1930, 1842, 715, 1929, 200, 1782, 1864]
show_click_filter_slots = show_click_filter_slots[:showclick_filter_topK]
show_click_filter_slots = list(hadamard_slot) + hash_slots + show_click_filter_slots
show_click_slots = [slot_id for slot_id in dnn_slots if slot_id not in show_click_filter_slots]
show_click_labels = ['show', 'click', 'skip', 'is_finish', 'ispraise', 'isshare', 'isreadcomment',
                     'iswritecomment', 'isfollow', 'isfeedback', 'stime', 'splaytime']
show_click_emb_dim = len(show_click_labels)
print("show_click_slots:", show_click_slots)
print("show_click_filter_slots:", show_click_filter_slots)

biz_slot = 2104
biz_weights = []
for x in biz_debias.strip().split('_'):
    sp = x.split(':')
    for biz in sp[0].split(','):
        biz_weights.append((int(biz), float(sp[1])))
print('biz_weights:', biz_weights)

label_configs = [['stime', 10000, 1, 1],
                 ['splaytime', 10000, 2, 2],
                 ['ispraise', 10000, 9, 9],
                 ['isshare', 10000, 10, 10],
                 ['isreadcomment', 10000, 11, 11],
                 ['iswritecomment', 10000, 12, 12],
                 ['isfollow', 10000, 13, 13],
                 ['isfeedback', 10000, 14, 14],
                 ]


class QBMiniFLoatModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.autoint_layer = AutoIntModel(prefix='mhsar',
                                          attention_embedding_size=embedding_dim * len(attention_slot_bias),
                                          num_heads=1,
                                          num_blocks=1,
                                          dropout_rate=0.2)
        self.bn_layer = tf.keras.layers.BatchNormalization(name="doc_input_layerbn")
        self.mmoe_model = MMoEModel(task_names=task_names,
                                    expert_num=4,
                                    expert_hidden_sizes=[512, 512, 256, 128],
                                    tower_hidden_sizes=[128, 1])
        self.show_click_layer = ShowClickLayer(slots=show_click_slots,
                                               filter_slots=show_click_filter_slots,
                                               replace_slot=replaced_slot,
                                               labels=show_click_labels,
                                               dim=show_click_emb_dim,
                                               task_names=task_names,
                                               norm_type=showclick_norm_type,
                                               dropout_rate=showclick_drop_rate,
                                               use_share=showclick_use_share,
                                               debug_mode=debug_mode,
                                               decay_weight=showclick_decay_weight,
                                               )
        self.denoise_net = {}
        for name in task_names:
            self.denoise_net[name] = DenoiseLayer(name=name)

    def call(self, inputs, training):
        shape = inputs['sparse_w'].shape
        old_dnn_feature = tf.reshape(inputs['sparse_w'], [-1, shape[-2] * shape[-1]])
        print("old_dnn_feature_shape:", old_dnn_feature.shape)
        sparse_features = [inputs[feature_name] for feature_name in new_feasign_feature_names] + [old_dnn_feature]
        attention_embedding_layer = tf.concat([inputs['seq_{}'.format(seq_name)] for seq_name in seq_names], axis=2)
        attention_output = self.autoint_layer(attention_embedding_layer, training)
        print("attention_output:", attention_output.shape)
        show_click_deep_out, show_click_wide_outs = self.show_click_layer(inputs, training)
        print("show_click deep_out shape:", show_click_deep_out.shape, "wide shape:", show_click_wide_outs[0].shape)
        if showclick_check:
            self.show_click_layer.check(inputs)
            show_click_check_emb = [inputs[f'show_click_api_{slot}'] for slot in show_click_slots]
            show_click_check_emb = tf.concat(show_click_check_emb, axis=-1) * 0.0
            print("show_click_check_emb:", show_click_check_emb.shape)
            # embedding_merge_layer = tf.concat([attention_output] + sparse_features + [show_click_check_emb], axis=1)
            embedding_merge_layer = tf.concat([attention_output] + sparse_features
                                              + [show_click_deep_out, show_click_check_emb], axis=1)
        else:
            embedding_merge_layer = tf.concat([attention_output] + sparse_features, axis=1)
            # embedding_merge_layer = tf.concat([attention_output] + sparse_features, axis=1)
        print("showclick_check:", showclick_check, "embedding_merge_layer:", embedding_merge_layer.shape)
        embedding_merge_layer = self.bn_layer(embedding_merge_layer, training)
        deep_outs = self.mmoe_model(embedding_merge_layer, training)
        wide_outs = [inputs["sparse_lr_{}".format(index)] for index in range(1, num_tasks + 1)]
        print("concat shape: ", wide_outs[0].shape, show_click_wide_outs[0].shape, deep_outs[0].shape)

        weights = []
        for biz, weight in biz_weights:
            weights.append(tf.where(inputs[f'biz_type_{biz}'] > 0, weight, 0.0))
        weights = tf.add_n(weights)
        weights = tf.where(tf.not_equal(weights, 0), weights, 1.0)
        # tf.print('biz weights:', weights)
        # if tf.math.reduce_any(weights != 1.0):
        #     tf.print('biz weights2:', weights, summarize=-1)

        # task outputs
        ratio_out = tf.add_n([wide_outs[0], show_click_wide_outs[0], deep_outs[0]])
        ratio_pred = tf.sigmoid(ratio_out)
        ratio_pred = tf.clip_by_value(ratio_pred * weights, clip_value_min=0.0, clip_value_max=1.0)
        skip_out = tf.add_n([wide_outs[1], show_click_wide_outs[1], deep_outs[1]])
        skip_pred = tf.sigmoid(skip_out)
        skip_pred = tf.clip_by_value(skip_pred / weights, clip_value_min=0.0, clip_value_max=1.0)
        finish_out = tf.add_n([wide_outs[2], show_click_wide_outs[2], deep_outs[2]])
        finish_pred = tf.sigmoid(finish_out)
        finish_pred = tf.clip_by_value(finish_pred * weights, clip_value_min=0.0, clip_value_max=1.0)

        denoise_net_logits = {}
        for name in task_names:
            lr_input = inputs["weightnet_sparse_lr_{}".format(name)]
            denoise_net_logits[name] = self.denoise_net[name]((embedding_merge_layer, lr_input), training)

        out_sum = tf.concat([ratio_pred, finish_pred, skip_pred], axis=1)
        predict_new = tf.concat([ratio_pred, finish_pred, skip_pred], axis=1)
        predict = tf.concat([ratio_pred, finish_pred, skip_pred], axis=1)

        return {'ratio_out': ratio_out,
                'skip_out': skip_out,
                'finish_out': finish_out,
                'ratio_pred': ratio_pred,
                'skip_pred': skip_pred,
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
        self.set_optimizer(tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, learning_rate=args.dnn_lr))
        self.dnn = QBMiniFLoatModel()
        self.dist_show_click_grad = {}
        for replica_id in range(tf.distribute.get_strategy().num_replicas_in_sync):
            self.dist_show_click_grad["GPU:{}".format(replica_id)] = {}
        self.global_step = tf.Variable(1.0, trainable=False, name="global_step")

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
        ftrl_optimizer = numerous.optimizers.Ftrl(alpha=0.04, beta=1.0, lambda1=2.0, lambda2=1.0)
        emb_initializer = numerous.initializers.Uniform(minval=-0.001, maxval=args.emb_lr)
        emb_optimizer = numerous.optimizers.Adam(learning_rate=0.0001)
        emb_combiner = numerous.layers.Combiner.SUM
        # show click的初始化器、优化器与常规embedding有区别，我们使用GlobalSum来实现show click的累加
        show_click_initializer = numerous.initializers.RandomUniform(minval=0, maxval=0)
        show_click_optimizer = numerous.optimizers.GlobalSum(summary_decay_rate=showclick_decay_rate)
        show_click_combiner = numerous.layers.Combiner.SUM
        inputs = Inputs()
        double_hashing_config = numerous.layers.DoubleHashingCompressConfig(hash_range=1 << 26)

        for slot_id in hash_slots:
            feature_name = "sparse_w_{}".format(slot_id)
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=str(slot_id),
                embedding_dim=embedding_dim,
                initializer=emb_initializer,
                table_name=feature_name,
                combiner=emb_combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.int4,
                double_hashing_config=double_hashing_config
            )

        inputs.add_share_embedding_features(
            feature_names=["sparse_w_{}".format(slot_id) for slot_id in dnn_slots],
            slot_ids=dnn_slots,
            embedding_dim=embedding_dim,
            table_name="sparse_w",
            lengths=None,
            # merge_output=True,
            combiner=emb_combiner,
            optimizer=emb_optimizer,
            initializer=emb_initializer,
            dump_dtype=numerous.int8
        )

        for slot_id in new_feasign_slots:
            inputs.add_sparse_feature(
                feature_name=f"sparse_w_{slot_id}",
                slot_id=slot_id,
                embedding_dim=embedding_dim,
                combiner=emb_combiner,
                optimizer=emb_optimizer,
                initializer=emb_initializer,
                dump_dtype=numerous.int8
            )

        for i in range(len(attention_slot_bias)):
            bias = attention_slot_bias[i]
            target_slot = target_slots[i]
            seq_name = seq_names[i]
            seq_slots = [target_slot] + [slot + bias for slot in attention_item_slots]
            inputs.add_share_embedding_features(
                feature_names=["seq_{}_{}".format(seq_name, slot) for slot in seq_slots],
                slot_ids=seq_slots,
                embedding_dim=embedding_dim,
                table_name="seq_{}".format(seq_name),
                lengths=None,
                merge_output=True,
                combiner=emb_combiner,
                optimizer=emb_optimizer,
                initializer=emb_initializer,
                dump_dtype=numerous.int8
            )

        # 所有embedding使用无量原始的show click，用于后续验证多目标show click的正确性
        for slot_id in filter(lambda slot: slot not in show_click_filter_slots, show_click_slots):
            if showclick_check:
                inputs.add_sparse_feature(
                    feature_name=f'show_click_api_{slot_id}',
                    slot_id=slot_id,
                    embedding_dim=embedding_dim,
                    optimizer=emb_optimizer,
                    initializer=emb_initializer,
                    combiner=emb_combiner,
                    is_with_show_click=True
                )
            # 手动定义show_click table，1个show，n个目标，合计n+1维。
            if not showclick_use_share:
                inputs.add_sparse_feature(
                    feature_name=f'show_click_handcraft_{slot_id}',
                    slot_id=slot_id,
                    embedding_dim=show_click_emb_dim,
                    optimizer=show_click_optimizer,
                    initializer=show_click_initializer,
                    combiner=show_click_combiner
                )

        if showclick_use_share:
            merge_output = False if showclick_check else True
            show_click_feature_names = [f"show_click_handcraft_{slot_id}" for slot_id in show_click_slots]
            # # 手动定义show_click table，1个show，n个目标，合计n+1维。
            inputs.add_share_embedding_features(
                feature_names=show_click_feature_names,
                slot_ids=show_click_slots,
                embedding_dim=show_click_emb_dim,
                table_name="show_click_handcraft_w",
                lengths=None,
                combiner=show_click_combiner,
                optimizer=show_click_optimizer,
                initializer=show_click_initializer,
                merge_output=merge_output,
            )

        for index in range(1, num_tasks + 1):
            inputs.add_sparse_feature(
                feature_name="sparse_lr_{0}".format(index),
                slot_id=lr_slots,
                embedding_dim=1,
                initializer=emb_initializer,
                table_name="sparse_lr_{0}".format(index),
                combiner=emb_combiner,
                optimizer=ftrl_optimizer,
                zero_filter_threshold=1e-6,
                dump_dtype=numerous.float16
            )

        for name in task_names:
            inputs.add_sparse_feature(
                feature_name="weightnet_sparse_lr_{}".format(name),
                slot_id=lr_slots,
                embedding_dim=1,
                initializer=emb_initializer,
                table_name="weightnet_sparse_lr_{}".format(name),
                combiner=emb_combiner,
                optimizer=ftrl_optimizer,
                zero_filter_threshold=1e-6,
                dump_dtype=numerous.float16
            )

        for biz, w in biz_weights:
            inputs.add_dense_feature(feature_name='biz_type_{}'.format(biz),
                                     slot_id=biz_slot,
                                     lower_bound=biz, upper_bound=biz)

        for task, slot_id, lower_bound, upper_bound in label_configs:
            inputs.add_label(label_name=task, slot_id=slot_id,
                             lower_bound=lower_bound, upper_bound=upper_bound)

        return inputs.build()

    def call(self, inputs, training):
        inputs = self.preprocess_inputs(inputs)  # process the features into a dictionary
        # hadamard
        for slot_id in hadamard_slot:
            inputs['sparse_w_' + str(slot_id)] = inputs['sparse_w_' + str(hadamard_slot[slot_id][0])] * inputs[
                'sparse_w_' + str(hadamard_slot[slot_id][1])]
            print('hadamard_slot ', inputs['sparse_w_' + str(slot_id)].get_shape(),
                  inputs['sparse_w_' + str(hadamard_slot[slot_id][0])].get_shape())

        outputs = self.dnn(inputs, training)

        new_global_step = tf.identity(self.global_step) + 1.0
        inc_global_step_op = self.global_step.assign(new_global_step)
        tf.compat.v1.add_to_collections(tf.compat.v1.GraphKeys.UPDATE_OPS, inc_global_step_op)

        return outputs

    # 手动替换show click table的梯度
    def filter_gradients(self, grads_and_vars):
        filtered_grads_and_vars = []
        for grad, var in grads_and_vars:
            if "show_click_handcraft_" in var.name:
                grad = self.dist_show_click_grad[self.device]
                filtered_grads_and_vars.append((grad, var))
            else:
                filtered_grads_and_vars.append((grad, var))
            if print_interval > 0 and self.global_step % print_interval == 0 and \
                    "show_click_handcraft_" in var.name:
                tf.print("global step:", self.global_step,
                         'var.name:', var.name, 'var:', var, 'after grad:', grad)
            # if tf.math.reduce_any(tf.math.is_nan(grad)):
            #     tf.print('var_grad is nan:', var, grad, summarize=-1)
        # 梯度裁剪
        # grads_and_vars = [(tf.where(tf.math.is_nan(g), tf.zeros_like(g), g), v) for g, v in grads_and_vars]
        # return grads_and_vars
        return filtered_grads_and_vars

    def compute_loss(self, labels, model_outputs, sample_weights, training):
        ratio_pred = model_outputs['ratio_pred']
        skip_pred = model_outputs['skip_pred']
        finish_pred = model_outputs['is_finish_pred']
        skip_out = model_outputs['skip_out']
        finish_out = model_outputs['finish_out']
        denoise_net_ratio_logit = model_outputs['denoise_net_ratio_logit']
        denoise_net_skip_logit = model_outputs['denoise_net_skip_logit']
        denoise_net_finish_logit = model_outputs['denoise_net_finish_logit']

        splaytime = tf.clip_by_value(labels["splaytime"], clip_value_min=0.0, clip_value_max=3600)
        stime = tf.clip_by_value(labels["stime"], clip_value_min=0.0, clip_value_max=3600)
        # ratio label
        ratio_label = splaytime / (stime + 1e-8)
        ratio_label = tf.clip_by_value(ratio_label, 0.0, 1.0, name="ratio_label_clip")
        ratio_label = tf.cast(ratio_label, tf.float32, name="ratio_label")

        # skip label
        sPlaytime_shape = tf.fill(tf.shape(splaytime), 5.0)
        skip_label = tf.less(splaytime, sPlaytime_shape)
        skip_label = tf.cast(skip_label, tf.float32, name="skip_label")
        # if showclick_check:
        #     labels['label'] = tf.cast(tf.less(splaytime, sPlaytime_shape), tf.float32, name="skip_label")
        #     skip_label = labels['label']

        # finish label
        finish_label = tf.less_equal(stime, splaytime)
        finish_label = tf.cast(finish_label, tf.float32, name="finish_label")

        action_sum = tf.add_n([labels["ispraise"], labels["isshare"], labels["isreadcomment"],
                               labels["iswritecomment"], labels["isfollow"], labels["isfeedback"]])
        noise_sample_flag = tf.logical_and(
            tf.logical_and(labels["stime"] == labels["splaytime"], labels['stime'] > 0), action_sum == 0)
        noise_sample_flag = tf.cast(noise_sample_flag, tf.float32, name="noise_sample_flag")
        # noise_sample_flag = labels["noise_sample_flag"]

        weightnet_sample_mask = tf.reshape(noise_sample_flag < 0.5, [-1])

        denoise_net_ratio_label = ratio_label
        denoise_net_ratio_label = tf.boolean_mask(denoise_net_ratio_label, weightnet_sample_mask)
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

        # skip denoise net
        denoise_net_skip_label = tf.boolean_mask(skip_label, weightnet_sample_mask)
        denoise_net_skip_logit_mask = tf.boolean_mask(denoise_net_skip_logit, weightnet_sample_mask)

        denoise_net_skip_loss = tf.cond(tf.equal(tf.size(denoise_net_skip_label), 0),
                                        lambda: tf.zeros_like(denoise_net_skip_label),
                                        lambda: tf.nn.sigmoid_cross_entropy_with_logits(
                                            logits=denoise_net_skip_logit_mask,
                                            labels=denoise_net_skip_label))
        denoise_net_skip_pred = tf.sigmoid(denoise_net_skip_logit, "denoise_net_skip_pred")
        denoise_net_skip_weight = tf.reshape(
            tf.stop_gradient(denoise_net_skip_pred, name="denoise_net_skip_weight1"), [-1])
        denoise_net_skip_weight = tf.where(weightnet_sample_mask, tf.zeros_like(denoise_net_skip_weight, tf.float32),
                                           denoise_net_skip_weight, name="denoise_net_skip_weight2")

        # finish denoise net
        denoise_net_finish_label = finish_label
        denoise_net_finish_label = tf.boolean_mask(denoise_net_finish_label, weightnet_sample_mask)
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

        ratio_loss = tf.math.square(ratio_pred - ratio_label) * tf.reshape(denoise_net_ratio_weight, [-1, 1])
        skip_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=skip_out, labels=skip_label) * tf.reshape(1 - denoise_net_skip_weight, [-1, 1])
        finish_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=finish_out, labels=finish_label) * tf.reshape(denoise_net_finish_weight, [-1, 1])

        # show_click_labels
        show_click_grads = [tf.ones_like(splaytime), labels['label'],
                            skip_label, finish_label, labels["ispraise"],
                            labels["isshare"], labels["isreadcomment"], labels["iswritecomment"],
                            labels["isfollow"], labels["isfeedback"], stime, splaytime]
        show_click_grads = tf.cast(tf.concat(show_click_grads, axis=1), dtype=tf.float32)
        if showclick_use_share:
            show_click_grads = tf.tile(tf.expand_dims(show_click_grads, axis=1), [1, len(show_click_slots), 1])
        self.dist_show_click_grad[self.device] = show_click_grads
        # tf.print("dist_show_click_grad:", self.dist_show_click_grad[self.device])

        final_loss = [ratio_loss, skip_loss, finish_loss, denoise_net_ratio_loss, denoise_net_skip_loss,
                      denoise_net_finish_loss]
        # process will block
        # tmp_loss = tf.concat(final_loss, axis=-1)
        # # tf.print('tmp_loss is nan:', tf.math.reduce_any(tf.math.is_nan(tmp_loss)))
        # if tf.math.reduce_any(tf.math.is_nan(tmp_loss)):
        #     tf.print('loss is nan:', tmp_loss, summarize=-1)

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
        return final_loss


numerous.cluster.start(numerous.cluster.get_cluster_config(run_mode=numerous.cluster.RunMode.CLUSTER))
with numerous.distribute.get_strategy().scope():
    model = DNNModel(name="qbminifloat")
    model.init_by_run_config(RunConfig(GPU_CONFIG))
    model.run()
numerous.cluster.stop()
