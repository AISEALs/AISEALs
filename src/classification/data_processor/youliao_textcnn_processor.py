from common_struct.common_class import ModeKeys
from data_processor.data_processor import BaseDataProcessor

'''
raw data: 158:/opt/youliao_syn/data 
grep "zjb-"  这样可以过滤出来
'''

class YouliaoTextCNNProcessor(BaseDataProcessor):
    """Processor for the youliao data set."""
    def __init__(self, base_dir, task_name="youliao_text_cnn", task_id="0", use_hdfs=False, mode=ModeKeys.TRAIN):
        super(YouliaoTextCNNProcessor, self).__init__(base_dir, task_name, task_id, use_hdfs)
        self.use_hdfs = use_hdfs
        self.mode = mode

    #todo :支持predict
    def convert_line2example(self, line):
        from data_prepare import youliao_raw_data_etl
        return youliao_raw_data_etl.convert_line2example(line, self.use_hdfs)

    def convert_example2tfexample(self, line, ex_index=None):
        def parse_line(sample, vocab):
            # 文本转为词ID序列，未知或填充用的词ID为0
            id_documents = list(vocab.transform([sample]))
            return id_documents[0]

        from data_processor.data_processor import Example
        e = Example.deserialize_from_str(line)
        # example = InputExample(guid=e.guid, text_a=e.line, text_b=None, label=e.label)
        from protobuf import pb2_Feature, pb2_Features, pb2_Int64List, pb2_Example
        features = {}
        if self.mode != ModeKeys.PREDICT:
            label_list = self.get_labels()
            features["label"] = pb2_Feature(int64_list=pb2_Int64List(
                value=[label_list.index(e.label)]))
        #这里必须把np.int64 转成int，不然后面反序列化会出错
        ink = parse_line(e.line.replace('\x001', ' '), self.get_vocab())
        tmp = [int(i) for i in ink]
        features["feature"] = pb2_Feature(int64_list=pb2_Int64List(value=tmp))
        f = pb2_Features(feature=features)
        example = pb2_Example(features=f)
        return example

