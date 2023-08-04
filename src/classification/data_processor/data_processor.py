#encode=utf8
import os
import csv
from tools import tools


class OneCateFileInfo(object):
    def __init__(self, cate_data_dir, label, total_num):
        self.label = label
        self.total_num = total_num if total_num else 0
        self.train_num = 0
        self.eval_num = 0
        # for local mode
        self.file_name = os.path.join(cate_data_dir, '{r}.tsv'.format(r=label))
        self.writer = None
        self.reader = None

    def filter_and_calculate(self):
        if self.total_num <= 10:
            return False
        self.eval_num = int(self.total_num * 0.1)
        if self.eval_num > 1024:
            self.eval_num = 1024
        self.train_num = self.total_num - self.eval_num
        return True

    def flush(self):
        if self.writer:
            self.writer.flush()
    #-----------------------------------------------------------------
    # for local mode
    def write_line(self, line):
        if not self.writer:
            self.writer = open(self.file_name, 'w+', encoding='UTF-8')
        self.writer.write(line + '\n')
        self.total_num += 1

    def get_reader(self):
        if not self.reader:
            self.reader = open(self.file_name, 'r+', encoding='UTF-8')
        return self.reader

    def get_df(self):
        import pandas as pd
        doc_df = pd.read_csv(self.file_name, sep='\t', header=None, encoding='utf-8')
        doc_df.columns = ["guid", "cuid", "label", "line"]
        return doc_df
    #-----------------------------------------------------------------


class State(object):
    def __init__(self):
        self.total_num = 0
        self.train_num = 0
        self.eval_num = 0

    def update(self, total_num, train_num, eval_num):
        self.total_num += total_num
        self.train_num += train_num
        self.eval_num += eval_num

    def clear(self):
        self.total_num = 0
        self.train_num = 0
        self.eval_num = 0



class FileHandlers(object):
    def __init__(self, task_dir, cate_data_dir, use_hdfs):
        self.file_infos = {}
        self.task_dir = task_dir
        self.class_file = os.path.join(task_dir, "label_classes.csv")
        self.cate_data_dir = cate_data_dir
        self.labels = []
        self.status = State()
        self.use_hdfs = use_hdfs

    def save_label_classes(self):
        data = []
        file_infos = filter(lambda x: x.filter_and_calculate(), self.file_infos.values())
        total_num = sum(map(lambda x: x.total_num, file_infos))

        for (label, file_info) in self.file_infos.items():
            if file_info.filter_and_calculate():
                data.append([str(label), str(file_info.total_num), str(file_info.train_num), str(file_info.eval_num), str("%.3f%%" % (file_info.total_num/total_num*100))])
        print("labels data: {}\n path: {}".format(str(data), self.class_file))

        if self.use_hdfs:
            client = tools.get_hdfs_client()
            # f.write(xxx), xxx: is bytes-like, not str
            with client.write(self.class_file) as f:
                f.write(','.join(["label", "total_num", "train_num", "eval_num", '\n']).encode('utf8'))
                for line in data:
                    line.append("\n")
                    f.write(','.join(line).encode('utf8'))
        else:
            with open(self.class_file, "w", encoding='utf8') as f:
                writer = csv.writer(f)
                writer.writerow(["label", "total_num", "train_num", "eval_num", "percent"])
                for line in data:
                    writer.writerow([s for s in line])
        for f in self.file_infos.values():
            f.flush()

    def load_label_classes(self, state):
        if len(self.labels) > 0:
            return
        data = []
        if self.use_hdfs:
            client = tools.get_hdfs_client()
            with client.read(self.class_file, encoding='UTF-8') as f:
                for line in f:
                    sp = line.strip().split(',')
                    d = dict(label=sp[0], total_num=sp[1], train_num=sp[2], eval_num=sp[3], percent=sp[4])
                    data.append(d)
        else:
            with open(self.class_file, mode='r', encoding='UTF-8') as f:
                reader = csv.DictReader(f)
                for line in reader:
                    data.append(line)
                print("-"*10 + "labels" + "-"*10)
                for i in data:
                    print(i)
        for row in data:
            try:
                label = row['label']
                if label.startswith("#") or label == "label":
                    continue
                info = OneCateFileInfo(self.cate_data_dir, label, int(row['total_num']))
                info.total_num = int(row['total_num'])
                info.train_num = int(row['train_num'])
                info.eval_num = int(row['eval_num'])
                self.file_infos[label] = info
                self.labels.append(label)
                state.update(info.total_num, info.train_num, info.eval_num)
            except Exception as e:
                print("load_label_classes " + str(e))
                continue
        print("load class file over, num: {} labels: {}".format(len(self.labels), str(self.labels)))

    def update_cate_file_info(self, label, total_num=None):
        if label not in self.file_infos:
            self.file_infos[label] = OneCateFileInfo(cate_data_dir=self.cate_data_dir, label=label, total_num=total_num)
            self.labels.append(label)
            print("generate file cate:{label} num:{num}".format(label=label, num=total_num))
            print("file_infos size: {}".format(len(self.file_infos)))

    # todo: 是不是该提出去
    def write_example(self, label, line):
        self.update_cate_file_info(label)
        if not os.path.exists(self.cate_data_dir):
            os.makedirs(self.cate_data_dir)
        self.file_infos[label].write_line(line)

    def get_file_handler(self, label):
        if len(self.file_infos) == 0:
            self.load_label_classes(self.status)
        return self.file_infos[label]


class Example(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, label, line, guid=0, cuid=0):
        # label是文件名字，不能包含/
        self.label = label.replace("/", "_")
        self.line = line
        self.guid = guid            # 全局唯一ID（10w*label_index + cuid）
        self.cuid = cuid            # 每个类别uid（Category Unique Identifier)

    def serialize_for_tsv(self):
        return '\t'.join([str(self.guid), str(self.cuid), self.label, self.line])

    @staticmethod
    def deserialize_from_str(line):
        try:
            sp = line.split('\t')
            return Example(guid=int(sp[0]), cuid=int(sp[1]), label=sp[2], line=sp[3])
        except:
            if len(line) >= 10:
                print("ERROR line: {}".format(line))
            return None


""" 
base_dir:
├── ${task_name}_raw_data        ##每个task_name原始目录只有一份
│   ├── 2018-09-13
│   └── 2018-09-14
└── ${task_name_task_id}         ##task_name下，会有多分生成数据
    ├── label_classes.csv
    ├── cate_data
    │   ├── 舞蹈.tsv              ##hdfs:part-00000,part-00001,...
    │   └── 科技.tsv
    └── tfrecord
        ├── eval.tfrecord-00000-of-00005
        └── train.tfrecord-00000-of-00005
"""


class BaseDataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, base_dir, task_name, task_id, use_hdfs, multiple=True, maps=None, label_name=""):
        self.base_dir = base_dir
        self.raw_data_dir = os.path.join(base_dir, "{s}_raw_data/".format(s=task_name))
        self.task_dir = os.path.join(base_dir, "{s}_{t}/".format(s=task_name, t=task_id))
        self.cate_data_dir = os.path.join(self.task_dir, "cate_data")
        self.tfrecord_dir = os.path.join(self.task_dir, "tfrecord")

        self.__file_handlers = FileHandlers(self.task_dir, self.cate_data_dir, use_hdfs)
        self.use_hdfs = use_hdfs
        #词库，可选项
        self.vocab = None
        # 标签映射
        self.multiple = multiple
        self.maps = maps
        self.label_name = label_name
        self.mapping_list = None

    '''
    每个继承实现，只需实现下面两个函数：
    1. 原始数据转换成 CateExample
    2. CateExample转换成tf.Example
    '''
    # -----------------------------------------------------------------
    # todo: 这里可以把CateExample抽象出一个基类, 提供（反）序列化机制，这样就不限于文本分类问题
    def convert_line2example(self, line):
        """把一行原始数据(raw_data) 解析成 一行分好类别的数据(Example)
        Args:
          line: 一行原始数据
        Returns:
          Example{
            label: 一个字符串，代表所属类别(可能为None)
            line: 处理好的一行数据(可能为None)
            guid: 不用管，不用传            #全局唯一ID（10w*label_index + cuid）
            cuid: 不用管，不用传            #每个类别uid（Category Unique Identifier)
          }
          或者 None
        """
        raise NotImplementedError()

    def convert_example2tfexample(self, line):
        """把一行分好类别的数据 解析成 tf.train.Example
        Args:
          line: str, 一行分好类别的数据
        Returns:
          tf.Example: local下为tf.train.Example, pyspark下为pb2_example.Example
          或者 None
        """
        raise NotImplementedError()

    # -----------------------------------------------------------------
    def store_vocab(self):
        from text_classification.data_prepare import vocabulary_processor
        vocabulary_processor.gen_local_vocab(self)

    def load_vocab(self):
        from text_classification.data_prepare import vocabulary_processor
        self.vocab = vocabulary_processor.restore_vocab(self, self.use_hdfs)
        print("-"*10 + "vocab" + "-"*10)
        print("load vocab size: {}".format(len(self.vocab.vocabulary_)))

    def get_vocab(self):
        if not self.vocab:
            self.load_vocab()
        return self.vocab

    def clear(self):
        self.__file_handlers.status.clear()

    def load_classes(self):
        self.__file_handlers.load_label_classes(self.__file_handlers.status)

    def get_labels(self):
        """Gets the list of labels for this data set."""
        labels = self.__file_handlers.labels
        if len(labels) == 0:
            self.load_classes()
        return labels

    # 标签映射
    def labels_mapping(self):
        labels = self.get_labels()
        if self.multiple:
            if self.maps is not None:
                labels_map = dict(map(lambda label: (label, self.maps[label]) if label in self.maps else (label, label), labels))
            else:
                labels_map = dict(map(lambda label: (label, label), labels))
        else:
            # 二分类时生效, label_name + "其他"
            if self.label_name not in labels:
                raise Exception("ERROR label_name:{} not in labels".format(self.label_name))
            labels_map =  dict(map(lambda x: (x, "其他") if x != self.label_name else (x, x), labels))
        return labels_map

    def label_mapping_list(self):
        if self.mapping_list is None:
            labels_map = self.labels_mapping()
            self.mapping_list = list(set(labels_map.values()))
        return self.mapping_list

    def get_file_handler(self, label):
        return self.__file_handlers.file_infos[label]

    def update_cate_file_info(self, label, num):
        self.__file_handlers.update_cate_file_info(label, num)

    def save_label_classes(self):
        self.__file_handlers.save_label_classes()

    def get_train_num(self):
        return self.__file_handlers.status.train_num

    def get_eval_num(self):
        return self.__file_handlers.status.eval_num