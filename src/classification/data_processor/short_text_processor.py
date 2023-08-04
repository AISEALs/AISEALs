import os
import traceback
from common_struct.common_class import ModeKeys
from data_processor.data_processor import BaseDataProcessor, Example


'''
raw data: 158:/opt/youliao_syn/data 
grep "zjb-"  这样可以过滤出来
'''

class ShortTextProcessor(BaseDataProcessor):
    """Processor for the youliao data set."""
    def __init__(self, base_dir, task_name="short_text", task_id="0", use_hdfs=False, mode=ModeKeys.TRAIN, use_seg=False, multiple=True, label_name=""):
        super(ShortTextProcessor, self).__init__(base_dir, task_name, task_id, use_hdfs, multiple=multiple, label_name=label_name)
        self.use_hdfs = use_hdfs
        self.mode = mode
        user_dict = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "tribe_labels/data/my_dict.txt"
        )
        if use_seg:
            import pkuseg
            self.seg = pkuseg.pkuseg(user_dict=user_dict)
        else:
            self.seg = None

    #todo :支持predict
    def convert_line2example(self, line):
        try:
            line = line.strip()
            if line == "":
                return None
            sp = line.split("\t")
            if self.mode == ModeKeys.TRAIN or self.mode == ModeKeys.EVAL or len(sp) >= 2:
                label = sp[0]
                if "__label__" in label:
                    label = label.replace("__label__", "")
                content = sp[1]
            else:
                label = ""
                content = sp[0]

            from common_tools import tokenization
            content = tokenization.convert_to_unicode(content)
            if self.seg != None:
                content = " ".join(self.seg.cut(content.replace(" ", "")))
            label = tokenization.convert_to_unicode(label)
            return Example(guid=0, cuid=0, label=label, line=content)
        except Exception:
            traceback.print_exc()
            print("error line: {}".format(line))
            return None

    def convert_example2tfexample(self, line, ex_index=None):
        from models.bert import bert_per_etl
        self_realize_func = bert_per_etl.file_based_convert_examples_to_features
        return self_realize_func(ex_index, line, self.label_mapping_list())

        # if self.vocab == None:
        #     self.load_vocab()
        #
        # def parse_line(sample, vocab):
        #     # 文本转为词ID序列，未知或填充用的词ID为0
        #     id_documents = list(vocab.transform([sample]))
        #     return id_documents[0]
        #
        # from data_processor.data_processor import Example
        # e = Example.deserialize_from_str(line)
        # if e is None:
        #     return None
        #
        # from protobuf import pb2_Feature, pb2_Features, pb2_Int64List, pb2_Example
        # features = {}
        # if self.mode != ModeKeys.PREDICT:
        #     label_list = self.get_labels()
        #     features["label"] = pb2_Feature(int64_list=pb2_Int64List(
        #         value=[label_list.index(e.label)]))
        # #这里必须把np.int64 转成int，不然后面反序列化会出错
        # ink = parse_line(e.line.replace('\x001', ' '), self.vocab)
        # tmp = [int(i) for i in ink]
        # features["feature"] = pb2_Feature(int64_list=pb2_Int64List(value=tmp))
        # f = pb2_Features(feature=features)
        # example = pb2_Example(features=f)
        # return example

