import os
import sys
sys.path.append("../")
from protobuf import dlserver_pb2
from protobuf import dlserver_pb2_grpc
import grpc
from structs import OutputData
from structs import AlgoId
from data_processor.processor_manager import get_processor
from common_struct.common_class import ModeKeys
from conf import config


# base_path = "/Users/jiananliu/AISEALs/data/text_classification"
# base_path = "/opt/AISEALs/data/text_classification"
base_path = os.path.abspath(os.path.join(config.root_dirname, "data/text_classification"))
task_name = "tribe_labels"
task_id = "20190708"

print("base_path:" + str(base_path))

processor = get_processor(base_path, task_name, task_id, use_hdfs=False, mode=ModeKeys.PREDICT, multiple=False,
                          label_name="心情_情绪_想法表达")


# host_port = '10.126.106.160:50051' if config.debug_mode else '10.126.106.210:50051'
host_port = '10.126.106.210:50051' # for debug because dl server is online
print(host_port)
channel = grpc.insecure_channel(host_port)

def trans2examples(train_data_list):
    def transform_line(train_data):
        cate_example = processor.convert_line2example(train_data.content)
        if not cate_example:
            return None
        cate_example_str = cate_example.serialize_for_tsv()
        tf_example = processor.convert_example2tfexample(cate_example_str)
        return tf_example

    examples = list(filter(lambda x: x != None, map(transform_line, train_data_list)))

    print(len(examples))
    example_list = dlserver_pb2.ExampleList(examples=examples)

    return example_list

def bert_predict(inputs, task_id):
    stub = dlserver_pb2_grpc.ExamplesDLServerStub(channel)
    print("-------------- GetScores --------------")
    example_list = trans2examples(inputs)
    dl_request = dlserver_pb2.DLRequest(task_id=str(task_id), example_list=example_list)
    outputs = stub.GetOutputs(dl_request)
    if not outputs:
        print("error request task_id:{} num:{} failed".format(task_id, len(inputs)))
        return []

    predicts = outputs.features.feature.get("predictions").int64_list.value
    label_list = processor.label_mapping_list()
    predicts = [label_list[i] for i in predicts]
    logits = outputs.features.feature.get("logits").float_list.value
    assert len(logits) == len(label_list) * len(inputs)
    step = len(label_list)
    logits_list = [logits[i:i+step] for i in range(0, len(logits),step)]
    outputs = []
    for input, logits, predict in zip(inputs, logits_list, predicts):
        if isinstance(logits, list):
            logits = ','.join(map(str, logits))
        outputs.append(OutputData(AlgoId.BERT, input.info_id, predict, logits))
    return outputs

if __name__ == '__main__':
    from structs import InputData
    def read_input_dataset():
        test_file = "../tribe_labels/data/test.csv"
        with open(test_file, 'r+', encoding='utf-8') as f:
            train_data_list = []
            total_count = 0
            for line in f:
                sp = line.split("\t")
                info_id = total_count
                label = sp[0]
                content = sp[1].strip()
                train_data_list.append(InputData(info_id, label, content))
                total_count += 1
                if len(train_data_list) >= 100:
                    yield train_data_list
                    train_data_list = []


    inputs_dataset = read_input_dataset()
    for inputs in inputs_dataset:
        results = bert_predict(inputs, 895)
        print(results)
        break

