# 更新时间: 2023年07月07日11:10:42
# 维护人: kyletu, zhequanzhou
# 版本V1

# 更新说明
# V1:
# 适合作为baseline配置
# 特点:
# 1. 根据cpu core数设置各个线程的数量.
# 2. 默认pass, group的设置方式为按照文件大小. 根据可用HbmMem和HostMem的大小, 自动缩放pass和group.
# 3. 常用参数都可以通过命令行方式获取,无需修改代码.


import sys
import os
import math
import numerous
from numerous.utils.config import get_config

def cpu_cores_per_gpu():
    pool_name = os.environ.get("ENV_POOL_NAME", "")
    if "-v100" in pool_name or "_v100" in pool_name:
        return 11
    elif "-a10" in pool_name or "_a10" in pool_name:
        return 12
    else:
        return 23

#TODO(zhequan):根据不同数据格式设置不同的ratio
#一个pass在HbmMem中占用率, 这边假设pass file size和pass 显存中的占用成比例.
#可以通过调整kPassFileSizeRatioInHbm来调整一个pass file size的大小
kPassFileSizeRatioInHbm=0.1
def HbmMem_per_gpu():
    pool_name = os.environ.get("ENV_POOL_NAME", "")
    if "-v100" in pool_name or "_v100" in pool_name:
        return 30
    elif "-a10" in pool_name or "_a10" in pool_name:
        return 22
    else:
        return 40

#一个group在HostMem中占用率, 这边假设group file size和group 内存中的占用成比例.
#可以通过调整kGroupFileSizeRatioInHost来调整一个group file size的大小
kGroupFileSizeRatioInHost=0.1
def HostMem_per_gpu():
    pool_name = os.environ.get("ENV_POOL_NAME", "")
    if "-v100" in pool_name or "_v100" in pool_name:
        return 45
    elif "-a10" in pool_name or "_a10" in pool_name:
        return 45
    else:
        return 128

def aviliable_core(coefficient: float = 1.0, extra_per_gpu: int = 0):
    total_core = (cpu_cores_per_gpu() + extra_per_gpu) * numerous.distribute.local_gpu_num()
    return max(1, math.ceil(total_core * coefficient))

def aviliable_HbmMem_MB(coefficient: float=1.0):
    mem_GB = HbmMem_per_gpu()* numerous.distribute.local_gpu_num() * coefficient
    return int(1024 * mem_GB)

def aviliable_HostMem_MB(coefficient: float=1.0):
    mem_GB = HostMem_per_gpu()* numerous.distribute.local_gpu_num() * coefficient
    return int(1024 * mem_GB)

def build_gpu_conf():
    print("building numerous gpu config...")
    is_auto_optimize = bool(int(get_config().get_option("userdefine.config.is_auto_optimize", default_value="0")))
    data_source = get_config().get_option("data_source", default_value="hdfs")
    skip_recover_nn = bool(int(get_config().get_option("userdefine.config.skip_recover_nn", default_value="0")))
    do_recover_thread_num = int(get_config().get_option("userdefine.config.do_recover_thread_num", default_value=aviliable_core()))
    data_format = get_config().get_option("userdefine.config.data_format", default_value="row")
    open_hotkey = int(get_config().get_option("userdefine.config.open_hotkey", default_value="0"))
    file_loader_thread_num = int(get_config().get_option("userdefine.config.file_loader_thread_num", default_value=aviliable_core(coefficient=1/3, extra_per_gpu=1)))
    file_parse_thread_num = int(get_config().get_option("userdefine.config.file_parse_thread_num", default_value=aviliable_core(coefficient=1/3, extra_per_gpu=1)))
    parse_file_chan_capacity = int(get_config().get_option("userdefine.config.parse_file_chan_capacity", default_value="0"))
    
    
    reserved_gpu_memory_gb = int(get_config().get_option("userdefine.config.reserved_gpu_memory_gb", default_value="4"))
    max_gpu_memory_occupied_percentage = float(get_config().get_option("userdefine.config.max_gpu_memory_occupied_percentage", default_value="0.8"))
    file_num_per_pass = int(get_config().get_option("userdefine.config.file_num_per_pass", default_value="-1"))
    pass_num_per_group = int(get_config().get_option("userdefine.config.pass_num_per_group", default_value="1"))
    pass_max_files_size = int(get_config().get_option("userdefine.config.pass_max_files_size", default_value=aviliable_HbmMem_MB(kPassFileSizeRatioInHbm)))
    group_max_files_size = int(get_config().get_option("userdefine.config.group_max_files_size", default_value=aviliable_HostMem_MB(kGroupFileSizeRatioInHost)))
    if group_max_files_size < pass_max_files_size:
        group_max_files_size = pass_max_files_size
    if file_num_per_pass != -1:
        pass_max_files_size = -1
        group_max_files_size = -1

    part_num = int(get_config().get_option("userdefine.config.part_num", default_value="2048"))
    hbm_max_size = int(get_config().get_option("userdefine.config.hbm_max_size", default_value="128"))
    hbm_size = int(get_config().get_option("userdefine.config.hbm_size", default_value="128"))
    do_persistence_thread_num = int(get_config().get_option("userdefine.config.do_persistence_thread_num", default_value=aviliable_core()))
    load_checkpoint_thread_num = int(get_config().get_option("userdefine.config.load_checkpoint_thread_num", default_value=aviliable_core()))
    embedding_persistence_background_thread_num = int(get_config().get_option("userdefine.config.embedding_persistence_background_thread_num", default_value="0"))

    pipeline_batch_num = int(get_config().get_option("userdefine.config.pipeline_batch_num", default_value="2"))
    read_ins_thread_num = int(get_config().get_option("userdefine.config.read_ins_thread_num", default_value=aviliable_core(coefficient=1/6)))
    convert_ins_thread_num = int(get_config().get_option("userdefine.config.convert_ins_thread_num", default_value=aviliable_core(coefficient=1/3)))
    compute_batch_thread_num = int(get_config().get_option("userdefine.config.compute_batch_thread_num", default_value="1"))

    streaming_model_dump_threads = int(get_config().get_option("userdefine.config.streaming_model_dump_threads", default_value="-1"))
    model_dump_interval = int(get_config().get_option("userdefine.config.model_dump_interval", default_value="5"))
    is_ignore_non_batch_ins = bool(int(get_config().get_option("userdefine.config.is_ignore_non_batch_ins", default_value="1")))
    extract_unique_key_thread_num = int(get_config().get_option("userdefine.config.extract_unique_key_thread_num", default_value=aviliable_core(coefficient=1 / 3)))
    merge_thread_num = int(get_config().get_option("userdefine.config.merge_thread_num", default_value=aviliable_core(coefficient=1 / 3)))
    write_fileins_thread_num = int(get_config().get_option("userdefine.config.write_fileins_thread_num", default_value=aviliable_core(coefficient=1 / 3)))
    write_unique_key_thread_num = int(get_config().get_option("userdefine.config.write_unique_key_thread_num", default_value=aviliable_core(coefficient=1 / 3)))

    prepare_pass_thread_num = int(get_config().get_option("userdefine.config.prepare_pass_thread_num", default_value=aviliable_core(coefficient=0.5)))
    build_pass_thread_num = int(get_config().get_option("userdefine.config.build_pass_thread_num", default_value=aviliable_core(coefficient=0.5)))
    build_group_thread_num = int(get_config().get_option("userdefine.config.build_group_thread_num", default_value=aviliable_core(coefficient=1, extra_per_gpu=1)))
    build_group_postprocess_thread_num = int(get_config().get_option("userdefine.config.build_group_postprocess_thread_num", default_value=aviliable_core(coefficient=1/3)))

    always_complete = bool(int(get_config().get_option("userdefine.config.always_complete", default_value="1")))
    complete_hour = get_config().get_option("userdefine.config.complete_hour", default_value="0")
    dir_format = get_config().get_option("userdefine.config.dir_format", default_value="%Y-%m-%d/%H")

    GPU_CONFIG = {
        "is_auto_optimize": is_auto_optimize,
        "reserved_gpu_memory_bytes": 1024 * 1024 * 1024 * reserved_gpu_memory_gb,
        "max_gpu_memory_occupied_percentage": max_gpu_memory_occupied_percentage,
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
            "is_ignore_non_batch_ins": is_ignore_non_batch_ins,
            "one_ins_max_fea_num": 2048,
            # 用户读取样本的线程数，该配置在配置时需要考虑两个因素，如果开大的话那么读取的线程数变多，读取数据速度会快，同时会带来内存增加的风险，
            # 如果我们的样本的每个文件都比较大，而且训练过程中出现了OOM，那么建议将该值减小。
            "file_loader_thread_num": file_loader_thread_num,
            # 用户读取样本的线程数, 针对带有预处理插件的列存
            "file_parse_thread_num": file_parse_thread_num,
            # 从用户的样本中抽取unique的key的线程数,默认的8 * GPU卡数
            "extract_unique_key_thread_num": extract_unique_key_thread_num,
            # 从多个unique key的线程中对抽取出的key进行merge,默认的4 * GPU卡数
            "merge_thread_num": merge_thread_num,
            # 将转化后的样本写入到ssd中的线程数, 默认的2 * GPU卡数
            "write_fileins_thread_num": write_fileins_thread_num,
            # 将抽取的unique的key写入到ssd中的线程数，默认的2 * GPU卡数
            "write_unique_key_thread_num": write_unique_key_thread_num,
            # 解析样本时使用的管道大小
            "parse_file_chan_capacity": parse_file_chan_capacity,
            "file_num_per_pass": file_num_per_pass,
            "pass_num_per_group": pass_num_per_group,
            "pass_max_files_size":1024 * 1024 * pass_max_files_size,
            "group_max_files_size":1024 * 1024 * group_max_files_size,
            "dir_format": dir_format,
            "data_format": 'text' if data_format in ['row','text', 'string'] else 'parquet',
            "sample_col_delimiter": "|",
            "sample_feature_delimiter": ";",
            "feature_section_delimiter": ":",
            "id_col_pos": 0,
            "label_col_pos": 1,
            "feature_col_pos": 2,
            "plugin_config": {},
        },
        "parameter_server_config": {
            "part_num": part_num,
            "hbm_max_key_num": 1024 * 1024 * hbm_max_size,
            "hbm_max_dynamic_byte_size": 1024 * 1024 * hbm_max_size * hbm_size,
            "enable_group_param_prep": True,

            "hbm_hashmap_load_factor": 0.75,
            "prepare_pass_thread_num": prepare_pass_thread_num,  # 预准备下一个pass的参数的时候的线程数
            "build_pass_thread_num": build_pass_thread_num,  # 构建一个pass的参数的时候的线程数
            "build_group_thread_num": build_group_thread_num,  # 构建一个group的参数的时候的线程数
            "build_group_postprocess_thread_num": build_group_postprocess_thread_num,  # 处理完一个group后的一些处理过程的线程数
            "do_persistence_thread_num": do_persistence_thread_num,  # 对sparse的参数做持久化的线程数
            "load_checkpoint_thread_num": load_checkpoint_thread_num,  # 加载sparse参数checkpoint的线程数
            "embedding_persistence_background_thread_num": embedding_persistence_background_thread_num,
            "use_parallel_optimizer": False,
            "delete_after_unseen_days": 7,  # 2
            "do_recover_thread_num": do_recover_thread_num,
            "is_use_parallel_optimizer": False,  # 要使用GlobalSum优化器，这个配置必须为False.
            "cow_clip": {"r": 1.0, "lower_bound": 1e-5},
        },
        "avengers_manager_base_config": {
            "pipeline_batch_num": pipeline_batch_num,  # 流水线中缓存的batch个数
            "read_ins_thread_num": read_ins_thread_num,  # 训练过程中用于从ssd中按照batch读取样本的线程数
            "convert_ins_thread_num": convert_ins_thread_num,  # 训练过程中将读取的原始的batch样本转化为gpu所需要的数据格式的线程数
            "compute_batch_thread_num": compute_batch_thread_num,  # 训练过程中gpu用于执行训练的线程数
        },
        "saver_config": {
            "model_dump_interval": model_dump_interval,  # hdfs模式下是多少个目录导出一次模型，cdmq模式下是多少个pass导出一次模型。
            "dump_hotkey_size": 2 * (1 << 30) if open_hotkey else 0,
            "always_complete": always_complete,
            "complete_hour": complete_hour,
            "streaming_model_dump_threads": streaming_model_dump_threads,
            "is_streaming_model_dump": True,
            "skip_recover_nn": skip_recover_nn,
            "recover_nn_by_name": True
        },
    }
    
    if data_source in ["kafka", "cdmq", "ckafka"]:
        cdmq_debug = numerous.utils.config.get_config().get_option("userdefine.config.cdmq_debug", default_value="")
        cdmq_max_poll_interval_ms = numerous.utils.config.get_config().get_option("userdefine.config.cdmq_max_poll_interval_ms", default_value="300000")
        cdmq_consumer_thread_num = int(numerous.utils.config.get_config().get_option("userdefine.config.cdmq_consumer_thread_num", default_value="1"))

        cdmq_interval_min = int(numerous.utils.config.get_config().get_option("userdefine.config.cdmq_interval_min", default_value="10"))
        cdmq_max_sample_num_per_pass = int(numerous.utils.config.get_config().get_option("userdefine.config.cdmq_max_sample_num_per_pass", default_value="491520"))
        cdmq_dump_max_interval_min = int(numerous.utils.config.get_config().get_option("userdefine.config.cdmq_dump_max_interval_min", default_value="30"))

        GPU_CONFIG['reader_config']['global_conf'] = {"max.poll.interval.ms": cdmq_max_poll_interval_ms}
        if cdmq_debug in ["generic", "broker", "topic", "metadata",
                           "feature", "queue", "msg", "protocol", "cgrp",
                           "security", "fetch", "interceptor", "plugin", "consumer",
                           "admin", "eos", "mock", "assignor", "conf", "all"]:
            GPU_CONFIG['reader_config']['global_conf']["debug"] = cdmq_debug
        GPU_CONFIG['reader_config']['cdmq_consumer_thread_num'] = cdmq_consumer_thread_num
        GPU_CONFIG['reader_config']['cdmq_interval_min'] = cdmq_interval_min
        GPU_CONFIG['reader_config']['cdmq_max_sample_num_per_pass'] = cdmq_max_sample_num_per_pass
        GPU_CONFIG['parameter_server_config']['is_serialized_cache'] = True
        GPU_CONFIG['saver_config']['model_dump_interval'] = int(cdmq_dump_max_interval_min / cdmq_interval_min)

    print("finish build numerous gpu config. GPU_CONFIG:", GPU_CONFIG, flush=True)
    return GPU_CONFIG