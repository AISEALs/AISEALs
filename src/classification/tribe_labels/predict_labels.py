#encoding=utf-8

import argparse
import pprint
import pickle
import pandas as pd
import sys

def read_rule_words(file_name):
    words = []
    with open(file_name, "r", encoding='utf-8') as f:
        for word in f:
            words.append(word.strip())
    print("read {} words count:{}".format(file_name, len(words)))
    return words


yali_words = read_rule_words("data/shenghuo_yali_words.txt")


def get_index_by_label(label):
    return label_classes.index(label)


def get_label_by_index(index):
    return label_classes[index]


def predict_labels(contents):
    if not isinstance(contents, list):
        contents = [contents]
    df = pd.Series(contents)

    if FLAGS.model_type == "fasttext":
        labels, probabs = model.precision(contents)
        labels = list(map(lambda x: x[0], labels))
        predicts = list(map(get_index_by_label, labels))
    else:
        x = tfidf_model.transform(df)  # 得到tf-idf矩阵，稀疏矩阵表示法
        predicts = logistic_model.precision(x)

    jingli_label = "生活经历"
    yali_label = "生活压力"
    if FLAGS.model_type == "fasttext":
        jingli_label = "__label__" + jingli_label
        yali_label = "__label__" + yali_label
    jingli_index = get_index_by_label(jingli_label)
    yali_index = get_index_by_label(yali_label)
    word_names_set = set(yali_words)
    def exists_yali_words(line):
        return len(set(line.split(' ')) & word_names_set) != 0

    predicts_df = pd.DataFrame(predicts, columns=['predict'])
    predicts_df.loc[:, 'exists'] = df.apply(exists_yali_words)

    def predict_by_rule(row):
        return yali_index if row['predict'] == jingli_index and row['exists'] else row['predict']

    y_predict = predicts_df.apply(predict_by_rule, axis=1)
    return y_predict.apply(get_label_by_index).tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--is_debug",
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help="use debug mode")
    parser.add_argument(
        "--model_type",
        type=str,
        default="lr",
        help="")
    parser.add_argument(
        "--model_version",
        type=str,
        default="1_1",
        help="")

    FLAGS, unparsed = parser.parse_known_args()
    pp = pprint.PrettyPrinter().pprint
    pp("FLAGS: " + str(FLAGS))
    pp("unparsed: " + str(unparsed))
    print(FLAGS.is_debug)

    if FLAGS.model_type == "fasttext":
        from fastText import load_model
        model = load_model("models/fasttext_model_{}.bin".format(FLAGS.model_version))
        label_classes = model.get_labels()
    else:
        tfidf_model = pickle.load(open("models/tfidf_model_{}.sav".format(FLAGS.model_version), 'rb'))
        label_encoder_model = pickle.load(open("models/lr_label_model_{}.sav".format(FLAGS.model_version), 'rb'))
        logistic_model = pickle.load(open("models/lr_model_{}.sav".format(FLAGS.model_version), 'rb'))
        label_classes = label_encoder_model.classes_.tolist()

    # content = "我   的   家乡   无锡   ，   那   是   个   人杰地灵   的   地方"
    # content = "身心   疲惫   啊   ！   累累   累累   累累   "
    # import pkuseg
    # seg = pkuseg.pkuseg(user_dict="data/my_dict.txt")
    # print(content.replace(" ", ""))
    # content = seg.cut(content.replace(" ", ""))
    # content = " ".join(content)
    # y_predict = predict_labels(content)
    # print(y_predict)
    contents = ["生活 太 难 ， 打工 不易 从 老家 来 快 20天 了", "2018 倒贴 了 10万 小时候 觉得 酒 很苦 ， 为什么 大人 们 还是 那么 爱喝 ， 长大 后 才 发现 ， 比起 生活 ， 酒 确实 甜 了 许多 。找对象   想 找 个 爱 我 的 女人 过完 下 半辈子"]
    y_predict = predict_labels(contents)
    print(y_predict)
    # data = open(sys.argv[1],'r')
    # correct = 0
    # for line in data:
    #   line = line.replace('\r','').replace('\n','').split('\t')
    #   label = line[0]
    #   if FLAGS.model_type != "fasttext":
    #     label = label.replace('__label__','')
    #   content = line[1]
    #   y_predict = predict_labels(content)
    #
    #   if label != y_predict[0]:
    #     print(label, content)
    #   else:
    #     correct += 1
    #
    # print(correct,104)
