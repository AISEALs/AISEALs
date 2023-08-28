import numerous

#################GPU训练配置#########################
# GPU_CONFIG全部配置参见: https://iwiki.woa.com/pages/viewpage.action?pageId=1524230667

GPU_CARD_NUM = numerous.distribute.local_gpu_num()
# sub_dir_file_regexes = ['.*part-00(0|1).*.gz', '.*part-00(2|3).*.gz', '.*part-00(4|5).*.gz', '.*part-00(6|7).*.gz',
#                         '.*part-00(8|9).*.gz']
# group_dir_regexs = [sub_dir_file_regexes[0]]

GPU_CONFIG = {
    "is_auto_optimize": False,
    "reserved_gpu_memory_bytes": 1024 * 1024 * 1024 * 4,
    "max_gpu_memory_occupied_percentage": 0.8,
    "reader_config": {
        "one_ins_max_fea_num": 2048,
        "is_ignore_zero_key": False,
        "is_ignore_negative_factor_key": False,
        "is_ignore_non_batch_ins": False,
        "file_loader_thread_num": 32 * GPU_CARD_NUM,
        "extract_unique_key_thread_num": 32 * GPU_CARD_NUM,
        "merge_thread_num": 8 * GPU_CARD_NUM,
        "write_fileins_thread_num": 4 * GPU_CARD_NUM,
        "write_unique_key_thread_num": 4 * GPU_CARD_NUM,
        "pass_dir_regexs": [".*"],
        # "sub_dir_file_regexs": sub_dir_file_regexes,
        # "group_dir_regexs": group_dir_regexs,       # 通过正则筛选需要训练哪些目录。
        "dir_format": "%Y-%m-%d/%H",
        "data_format": "text",
        "sample_col_delimiter": "|",
        "sample_feature_delimiter": ";",
        "feature_section_delimiter": ":",
        "id_col_pos": 0,
        "label_col_pos": 1,
        "feature_col_pos": 2,
        "plugin_path": "./libdefault_sample_preprocessor.so",
        "plugin_config": {},
    },
    "parameter_server_config": {
        "part_num": 1024,
        "hbm_max_key_num": 1024 * 1024 * 80,
        "hbm_max_dynamic_byte_size": 1024 * 1024 * 1024 * 20,
        "hbm_hashmap_load_factor": 0.75,
        "prepare_pass_thread_num": 12 * GPU_CARD_NUM,
        "build_pass_thread_num": 12 * GPU_CARD_NUM,
        "build_group_thread_num": 12 * GPU_CARD_NUM,
        "build_group_postprocess_thread_num": 4 * GPU_CARD_NUM,
        "do_persistence_thread_num": 24 * GPU_CARD_NUM,
        "load_checkpoint_thread_num": 24 * GPU_CARD_NUM,
        "use_parallel_optimizer": False,
        "delete_after_unseen_days": 7,
        "cow_clip": {"r": 1.0, "lower_bound": 1e-5}
    },
    "avengers_manager_base_config": {
        "pipeline_batch_num": 2,
        "read_ins_thread_num": 4 * GPU_CARD_NUM,
        "convert_ins_thread_num": 8 * GPU_CARD_NUM,
        "compute_batch_thread_num": 1,
    },
    "saver_config": {
        "model_dump_interval": 5,
        "recover_nn_by_name": True
    },
}

print("GPU_CONFIG:", GPU_CONFIG, flush=True)


def get_gpu_config():
    return GPU_CONFIG
