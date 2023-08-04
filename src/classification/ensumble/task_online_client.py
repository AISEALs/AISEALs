import os
import sys
sys.path.append("../")
from structs import InputData, OutputData
from model import *
from conf import config


def read_input_dataset(input_file):
    with open(input_file, 'r+', encoding='utf-8') as f:
        train_data_list = []
        total_count = 0
        for line in f:
            sp = line.split("\t")
            info_id = total_count
            label = sp[0]
            content = sp[1].strip()
            train_data_list.append(InputData(info_id, label, content))
            total_count += 1
            if len(train_data_list) >= 10:
                yield train_data_list
                train_data_list = []


def gen_features_by_file(mode):
    input_file = "./multiple4LR/data/{}.tsv".format(mode)
    output_file = "./{}.csv".format(mode)

    if os.path.exists(output_file):
        os.remove(output_file)

    inputs_dataset = read_input_dataset(input_file)

    debug_num = 0
    for inputs in inputs_dataset:
        debug_num += 1
        # if config.debug_mode and debug_num > 100:
        #     break
        fasttext_results = FaxttextModel().run(inputs)
        svm_results = SVMModel().run(inputs)
        rf_results = RFModel().run(inputs)
        # bert_results = BertModel(895).run(inputs) # 二分类模型，情绪-其他
        # for t in zip(fasttext_results, svm_results, rf_results, bert_results, inputs):
        for t in zip(fasttext_results, svm_results, rf_results, inputs):
            label = t[-1].label.replace("__label__", "")
            ilabel = Model.schema.index(label)
            features = ','.join(map(lambda output: output.features(), t[0:-1]))
            with open(output_file, "a") as f:
                f.write('{}\t{}\n'.format(ilabel, features))
                print('{}\t{}\t{}\n'.format(debug_num, ilabel, features))

    print("gen features file: {} success".format(output_file))


if __name__ == '__main__':
    for mode in ["train", "dev"]:
        gen_features_by_file(mode)

