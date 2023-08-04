import os
from data_processor.data_processor import BaseDataProcessor

'''
raw data: 158:/opt/youliao_syn/data 
grep "zjb-"  这样可以过滤出来
'''


class YouliaoProcessor(BaseDataProcessor):
    """Processor for the youliao data set."""
    def __init__(self, base_dir, task_name="youliao_bert", task_id="0", use_hdfs=False):
        super(YouliaoProcessor, self).__init__(base_dir, task_name, task_id, use_hdfs)

    def convert_line2example(self, line):
        from data_prepare import youliao_raw_data_etl
        return youliao_raw_data_etl.convert_line2example(line, self.use_hdfs)

    def convert_example2tfexample(self, line, ex_index=None):
        label_list = self.get_labels()
        from models.bert import bert_per_etl
        self_realize_func = bert_per_etl.file_based_convert_examples_to_features
        return self_realize_func(ex_index, line, label_list)


def test_lookup_data():
    # train_pattern = os.path.join(processor.tfrecord_dir, "eval.tfrecord*")
    # for test
    train_pattern = os.path.join(processor.tfrecord_dir, "*.tfrecord")
    print("train_pattern: {}".format(train_pattern))

    from models.bert import bert_per_etl
    train_input_fn = bert_per_etl.file_based_input_fn_builder(
        tfrecord_pattern=train_pattern,
        seq_length=128,
        is_training=True,
        drop_remainder=True)

    import tensorflow as tf
    dataset = train_input_fn(params={"batch_size": 1})
    with tf.Session() as sess:
        while True:
            try:
                x = sess.run([dataset])
                example = x[0]
                print("*** Example ***")
                # print("input_ids: %s" % " ".join([str(x) for x in example["input_ids"]]))
                # print("input_mask: %s" % " ".join([str(x) for x in example["input_mask"]]))
                # print("segment_ids: %s" % " ".join([str(x) for x in example["segment_ids"]]))
                print("label: (id = %d)" % example["label_ids"][0])
            except tf.errors.OutOfRangeError:
                break


def test_gen_line(processor):
    from protobuf.proto_tools import write_record
    i = 0
    from bert import bert_per_etl
    func = bert_per_etl.file_based_convert_examples_to_features
    with open("/tmp/test.txt", "wb+") as f2:
        with open("/Users/jiananliu/AISEALs/data/\
        text_classification/youliao_bert_cate_data/音乐.tsv", "r", encoding='UTF-8') as f:
            for line in f:
                example = processor.convert_example2tfexample(0, line, func)
                if example == None:
                    continue
                write_record(f2, example.SerializeToString())
                if i > 10:
                    break
                i += 1


if __name__ == '__main__':
    processor = YouliaoProcessor("/Users/jiananliu/AISEALs/data/text_classification/")
    # processor.etl_rawdata2catecsv()

    # processor.load_classes()
    # convert_features_func = functools.partial(processor.convert_examples2tfrecord,\
    #   output_shards=5, output_dir=None, func=file_based_convert_examples_to_features)
    # convert_features_func('train')
    # convert_features_func('eval')
    # convert_features_func('predict')

    test_gen_line(processor)
    test_lookup_data()
