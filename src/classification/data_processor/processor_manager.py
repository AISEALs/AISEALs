from common_struct.common_class import ModeKeys
from data_processor.short_text_processor import ShortTextProcessor
from data_processor.youliao_bert_processor import YouliaoProcessor
from data_processor.youliao_textcnn_processor import YouliaoTextCNNProcessor


processors = {
    "youliao_bert": YouliaoProcessor,
    "youliao_text_cnn": YouliaoTextCNNProcessor,
    "tribe_labels": ShortTextProcessor
}


def get_processor(base_dir, task_name, task_id, use_hdfs=False, mode=ModeKeys.TRAIN, multiple=True, label_name=""):
    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    return processors[task_name](base_dir, task_name=task_name, task_id=task_id, use_hdfs=use_hdfs, mode=mode, multiple=multiple, label_name=label_name)
