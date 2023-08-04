import argparse
import pprint
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# import scipy.sparse as sp
import pickle

import sys
import os
# 确保在命令行下执行，不出现ModuleNotFoundError: No module named ERROR
from tribe_labels.analysize_text_frequent_words import transform_features, get_index_by_label, gen_documents, get_tfidf_model, \
    get_label_encoder_model
from tribe_labels.predict_labels import read_rule_words

sys.path.append(os.getcwd())    # 在命令行中，把当前路径加入到sys.path中
# sys.path.append("/Users/jiananliu/work/python/AISEALs/text_classification")
print("sys.path:{}".format("\n".join(sys.path)))
from data_processor.processor_manager import get_processor


def train_lr(train_x, train_y):
    logistic_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

    # X1 shape [38287x38411] sparse matrix
    X_train_feature, Y_train_feature = transform_features(tfidf_model, label_encoder_model, train_x, train_y)

    logistic_model.fit(X_train_feature, Y_train_feature)
    return logistic_model


def shenghuo_cate_rule():
    jingli_label = "生活经历"
    jingli_index = get_index_by_label(label_encoder_model, jingli_label)
    yali_label = "生活压力"
    yali_index = get_index_by_label(label_encoder_model, yali_label)
    yali_words = read_rule_words("data/shenghuo_yali_words.txt")
    word_names_set = set(yali_words)

    def exists_yali_words(line):
        return len(set(line.split(' ')) & word_names_set) != 0

    predicts = logistic_model.precision(X_test_feature)
    predicts_df = pd.DataFrame(predicts, columns=['predict'])
    predicts_df.loc[:, 'exists'] = train_x.apply(exists_yali_words)

    def predict_by_rule(row):
        return yali_index if row['predict'] == jingli_index and row['exists'] else row['predict']

    y_predict = predicts_df.apply(predict_by_rule, axis=1)
    return y_predict


def save_model(model, file_name):
    pickle.dump(model, open(file_name, 'wb'))
    print("save models: {} success".format(file_name))


def run_score_by_label(test_x, test_y):
    test_x_y = test_x.copy().to_frame()
    test_x_y['label'] = test_y
    for label_df in test_x_y.groupby('label'):
        df = label_df[1]
        x_feature, y_feature = transform_features(tfidf_model, label_encoder_model, df['line'], df['label'])

        score = logistic_model.score(x_feature, y_feature)

        label_index = get_index_by_label(label_encoder_model, label_df[0])
        df.loc[:, 'proba'] = logistic_model.predict_proba(x_feature)[:, label_index]
        df.loc[:, 'ilabel'] = y_feature
        df.loc[:, 'ipredict'] = logistic_model.precision(x_feature)

        wrong_df = df[df['ilabel'] != df['ipredict']]
        last_wrong_df = wrong_df[['label', 'ilabel', 'ipredict', 'proba', 'line']]
        # last_wrong_df = pd.concat([df.loc[wrong_predict_indexes][['label', 'proba', 'line']].reset_index(drop=True), wrong_df[['ilabel', 'ipredict']].reset_index(drop=True)], axis=1)[['label', 'ilabel', 'ipredict', 'proba', 'line']]
        pd.set_option('display.width', 200)
        # pd.set_option('display.max_colwidth', 1000)
        # labels = label_encoder_model.classes_
        print("label:{} accuracy is: {}".format(label_df[0], score))
        print(last_wrong_df[last_wrong_df['proba'] < 0.2].sample_from_session(10).to_string(col_space=3, justify='start'))
        # print(last_wrong_df[last_wrong_df['proba'] < 0.2].sample(10).to_html())
        print('-'*40)
        # break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--is_debug",
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help="use debug mode")
    parser.add_argument(
        "--base_path",
        type=str,
        default="/Users/jiananliu/AISEALs/data/text_classification",
        help="refer to data_processor path tree")
    parser.add_argument(
        "--working_path",
        type=str,
        default="/Users/jiananliu/AISEALs/text_classification",
        help="refer to text_classification path")
    parser.add_argument(
        "--task_name",
        type=str,
        default="tribe_labels",
        help="task name")
    parser.add_argument(
        "--task_id",
        type=str,
        default="1",
        help="task id")

    FLAGS, unparsed = parser.parse_known_args()
    pp = pprint.PrettyPrinter().pprint
    pp("FLAGS: " + str(FLAGS))
    pp("unparsed: " + str(unparsed))
    print(FLAGS.is_debug)

    print("base_path:" + str(FLAGS.base_path))

    processor = get_processor(FLAGS.base_path, FLAGS.task_name, FLAGS.task_id, use_hdfs=False)

    docs_df = gen_documents(processor, FLAGS.is_debug)

    tfidf_model = get_tfidf_model(docs_df, FLAGS.working_path)

    label_encoder_model = get_label_encoder_model(docs_df)

    train_x, test_x, train_y, test_y = train_test_split(docs_df['line'], docs_df['label'], test_size=0.2)

    logistic_model = train_lr(train_x, train_y)

    X_test_feature, Y_test_feature = transform_features(tfidf_model, label_encoder_model, test_x, test_y)

    score = logistic_model.score(X_test_feature, Y_test_feature)
    print("total mean accuracy is: {}".format(score))

    save_model(logistic_model, 'models/lr_model_1_{}.sav'.format(FLAGS.task_id))
    save_model(label_encoder_model, 'models/lr_label_model_1_{}.sav'.format(FLAGS.task_id))
    save_model(tfidf_model, 'models/tfidf_model_1_{}.sav'.format(FLAGS.task_id))

    y_predict = shenghuo_cate_rule()

    rule_score = accuracy_score(Y_test_feature, y_predict)
    print("rule mean accuracy is: {}".format(rule_score))

    run_score_by_label(test_x, test_y)