import numerous
from numerous.utils.config import get_config
from dataclasses import dataclass

debug_mode = int(get_config().get_option("userdefine.config.debug_mode", default_value="0"))
debug_mode = bool(debug_mode)
pass_num = int(get_config().get_option("userdefine.config.pass_num", default_value="5"))
data_format = get_config().get_option("userdefine.config.data_format", default_value="column")
use_showclick = int(get_config().get_option("userdefine.config.use_showclick", default_value="0"))
use_showclick = bool(use_showclick)
skip_recover_nn = int(get_config().get_option("userdefine.config.skip_recover_nn", default_value="0"))
skip_recover_nn = bool(skip_recover_nn)
file_load_thread_per_gpu = int(get_config().get_option("userdefine.config.file_load_thread_per_gpu", default_value="8"))
dnn_lr = float(get_config().get_option("userdefine.config.dnn_lr", default_value="0.0001"))
emb_lr = float(get_config().get_option("userdefine.config.emb_lr", default_value="0.001"))
read_ins_thread_num = int(get_config().get_option("userdefine.config.read_ins_thread_num", default_value="4"))
convert_ins_thread_num = int(get_config().get_option("userdefine.config.convert_ins_thread_num", default_value="4"))
print("debug_mode: {}".format(debug_mode))
print("skip_recover_nn: {}".format(skip_recover_nn))
print("data_format: {}".format(data_format))
print("pass_num: {}".format(pass_num))
print("use_showclick: {}".format(use_showclick))
print("file_load_thread_per_gpu: {}".format(file_load_thread_per_gpu))
print("dnn_lr: {}".format(dnn_lr))
print("emb_lr: {}".format(emb_lr))
print("convert_ins_thread_num: {}".format(convert_ins_thread_num))
print("read_ins_thread_num: {}".format(read_ins_thread_num))

# showclick config
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
print("showclick_check: ", showclick_check)
print("showclick_norm_type: ", showclick_norm_type)
print("showclick_drop_rate: ", showclick_drop_rate)
print("showclick_use_share: ", showclick_use_share)
print("showclick_decay_rate: ", showclick_decay_rate)
print("showclick_decay_weight:", showclick_decay_weight)
print("showclick_filter_topK:", showclick_filter_topK)


def get_gpu_config():
    # sub_dir_file_regexes = ['part-.*{:01d}.parquet'.format(i) for i in range(10)]
    if pass_num == 10:
        sub_dir_file_regexes = ['.*part-.*{:01d}(.parquet)?'.format(i) for i in range(10)]
        group_dir_regexs = ['.*']
    else:
        sub_dir_file_regexes = ['.*part-{:01d}-.*'.format(i) for i in range(5)]
        group_dir_regexs = [sub_dir_file_regexes[0], sub_dir_file_regexes[3]]
        # group_dir_regexs = [sub_dir_file_regexes[0]]

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
            # 用户读取样本的线程数，该配置在配置时需要考虑两个因素，如果开大的话那么读取的线程数变多，读取数据速度会快，同时会带来内存增加的风险，
            # 如果我们的样本的每个文件都比较大，而且训练过程中出现了OOM，那么建议将该值减小。
            "file_loader_thread_num": file_load_thread_per_gpu * GPU_CARD_NUM,
            # 从用户的样本中抽取unique的key的线程数,默认的8 * GPU卡数
            "extract_unique_key_thread_num": 12 * GPU_CARD_NUM,
            "merge_thread_num": 8 * GPU_CARD_NUM,
            # 解析样本时使用的管道大小
            "parse_file_chan_capacity": 5,
            # 将转化后的样本写入到ssd中的线程数, 默认的2 * GPU卡数
            "write_fileins_thread_num": 4 * GPU_CARD_NUM,
            # 将抽取的unique的key写入到ssd中的线程数，默认的2 * GPU卡数
            "write_unique_key_thread_num": 4 * GPU_CARD_NUM,
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
            "hbm_max_key_num": 1024 * 1024 * 400,  # Pass FeaKeyCount
            "hbm_max_dynamic_byte_size": 1024 * 1024 * 1024 * 15,  # DynamicByteSize
            "hbm_hashmap_load_factor": 0.75,
            "prepare_pass_thread_num": 12 * GPU_CARD_NUM,
            "build_pass_thread_num": 12 * GPU_CARD_NUM,
            "build_group_thread_num": 12 * GPU_CARD_NUM,
            "build_group_postprocess_thread_num": 4 * GPU_CARD_NUM,
            "do_persistence_thread_num": 24 * GPU_CARD_NUM,
            "load_checkpoint_thread_num": 16 * GPU_CARD_NUM,
            "use_parallel_optimizer": False if use_showclick else True,
            "delete_after_unseen_days": 7,  # 2
            "do_recover_thread_num": 256,
            "cow_clip": {"r": 1.0, "lower_bound": 1e-5},
        },
        "avengers_manager_base_config": {
            "pipeline_batch_num": 2,
            "read_ins_thread_num": read_ins_thread_num * GPU_CARD_NUM,  # 训练过程中用于从ssd中按照batch读取样本的线程数
            "convert_ins_thread_num": convert_ins_thread_num * GPU_CARD_NUM,  # 训练过程中将读取的原始的batch样本转化为gpu所需要的数据格式的线程数
            "compute_batch_thread_num": 1,
        },
        "saver_config": {
            "model_dump_interval": 5,  # hdfs模式下是多少个目录导出一次模型，cdmq模式下是多少个pass导出一次模型。
            # "dump_hotkey_size": 2 * (1 << 30),
            "dump_hotkey_size": 0,
            "always_complete": True,
            "streaming_model_dump_threads": 128,
            "is_streaming_model_dump": True,
            "dump_file_num": 256,
            "skip_recover_nn": skip_recover_nn
        },
    }

    data_source = numerous.utils.config.get_config().get_option("data_source", default_value="hdfs")
    if data_source == 'kafka' or data_source == 'cdmq':
        cdmq_interval_min = 6
        GPU_CONFIG['reader_config']['cdmq_consumer_thread_num'] = 8
        GPU_CONFIG['reader_config']['cdmq_interval_min'] = cdmq_interval_min
        GPU_CONFIG['reader_config']['cdmq_max_sample_num_per_pass'] = 500000 * cdmq_interval_min
        GPU_CONFIG['reader_config']['plugin_path'] = "libonline_sample_preprocessor.so"
        GPU_CONFIG['saver_config']['model_dump_interval'] = 5
        GPU_CONFIG['saver_config']['always_complete'] = False
        GPU_CONFIG['saver_config']['complete_hour'] = '02,05,08,11,14,17,20,23'

    print("GPU_CONFIG:", GPU_CONFIG)
    return GPU_CONFIG


@dataclass
class CommonArgs:
    debug_mode: bool
    dnn_lr: float
    emb_lr: float
    showclick_check: bool
    showclick_norm_type: str
    showclick_drop_rate: float
    showclick_use_share: bool
    showclick_decay_rate: float
    showclick_decay_weight: float
    showclick_filter_topK: int


def get_common_args():
    args = CommonArgs(
        debug_mode=debug_mode,
        dnn_lr=dnn_lr,
        emb_lr=emb_lr,
        showclick_check=showclick_check,
        showclick_norm_type=showclick_norm_type,
        showclick_drop_rate=showclick_drop_rate,
        showclick_use_share=showclick_use_share,
        showclick_decay_rate=showclick_decay_rate,
        showclick_decay_weight=showclick_decay_weight,
        showclick_filter_topK=showclick_filter_topK
    )
    print("args:", args)
    return args
