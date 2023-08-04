import argparse
import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import json
import jieba
import sys
import os
# 确保在命令行下执行，不出现ModuleNotFoundError: No module named ERROR
print(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "../.."))    # 在命令行中，把当前路径加入到sys.path中
# print("sys.path:{}".format("\n".join(sys.path)))
from data_processor.processor_manager import get_processor
from common_tools import stop_word
from conf import config
from tools.tools import clean_html

max_num = 200

def compose2document(file_name, is_debug, num=None):
    contents = []
    with open(file_name, "r", encoding='utf8') as f:
        for line in f:
            if "recv from kafka data:" in line:
                mess = line.split("recv from kafka data:")[1]
                try:
                    mess_dict = json.loads(mess.replace("'", "\"").strip())
                except Exception as ex:
                    print("line: {}".format(line))
                    continue
                infoConent = mess_dict['infoContent'].replace("\n", "")
                contents.append(" ".join([clean_html(i) for i in jieba.cut(infoConent)]))

            if is_debug and len(contents) > 100:
                break
            if num is not None and len(contents) >= num:
                break

    return contents

#vim修改编码:set fileencoding=utf-8
def parse_jinghuatie(file_name):
    import pandas as pd
    doc_df = pd.read_csv(file_name, sep='\t', header=1, encoding='utf-8')
    doc_df.columns = ['time', 'id', 'user_id', 'title', 'content', 'labels', 'tribe_id', 'tribe_name', 'yewuxian_name']

    def func(line):
        return  " ".join([clean_html(i) for i in jieba.cut(line)])

    doc_df.loc[:, 'content'] = doc_df['content'].apply(func)
    return doc_df[['tribe_id', 'content']]
    # import pandas as pd
    # df = pd.DataFrame(file_name)
    # df.columns = ['time', 'id', 'user_id', 'title', 'content', 'labels', 'tribe_id', 'tribe_name', 'yewuxian_name']

def gen_documents(is_debug):
    # if is_debug:
    #     file_names_suffix = ['2019-08-10']
    # else:
    #     file_names_suffix = ['2019-10-1{}'.format(i) for i in range(7)]
    # file_name = lambda suffix: os.path.join(config.working_dirname, "online/log/finnal_predict_labels.log.{}".format(suffix))
    file_name = os.path.join(config.working_dirname, "tribe_labels/data/jinghuatie.txt")
    # docs_df = pd.concat([compose2document(file_name(suffix), is_debug) for suffix in file_names_suffix]).reset_index(drop=True)
    # return docs_df.sample(frac=1) #打乱顺序
    # lines = [i for suffix in file_names_suffix for i in compose2document(file_name(suffix), is_debug)]
    lines = parse_jinghuatie(file_name)
    return lines[['tribe_id', 'content']]


def gen_cate_documents(docs_df):
    cate2doc_ser = docs_df[['label', 'line']].groupby('label').apply(lambda df: ' '.join(df['line']))
    # pd.DataFrame({'Gene':s.index, 'count':s.values}), 两种方法
    cate2doc_df = pd.DataFrame(cate2doc_ser).reset_index()
    cate2doc_df.columns = ['label', 'line']
    return cate2doc_df

def gen_one_documents(docs_df):
    return " ".join(docs_df)

def get_tfidf_model(docs_df):
    stop_file_path = os.path.join(config.working_dirname, 'files/stop_words_tribe_labels.txt')
    stop_words = stop_word.load_stop_words(stop_file_path)
    tfidf_model = TfidfVectorizer(ngram_range=[1, 1], stop_words=stop_words, token_pattern=r"(?u)\b\w\w+\b").fit(docs_df)
    print("vocabulary size: {}".format(len(tfidf_model.vocabulary_)))
    return tfidf_model


def get_label_encoder_model(docs_df):
    label_encoder = LabelEncoder().fit(docs_df["label"])
    print("labels: {}".format(label_encoder.classes_))
    return label_encoder


def transform_features(tfidf_model, x):
    if not isinstance(x, list):
        x = list(x)
    x_ = tfidf_model.transform(x)  # 得到tf-idf矩阵，稀疏矩阵表示法
    print("vocab size: {}".format(len(tfidf_model.vocabulary_)))
    print("X matrix shape: {}".format(x_.shape))

    return x_


def print_matrix_by_view(topX, index_names, column_names, print_all=False):
    maxprint = topX.getmaxprint()

    A = topX.tocoo()

    # helper function, outputs "(i,j)  v"
    def tostr(row, col, data):
        indexes = map(lambda i: index_names[i], row)
        columns = map(lambda i: column_names[i], col)
        triples = zip(list(zip(indexes, columns)), data)
        return '\n'.join([('  %s\t%s' % t) for t in triples])

    if print_all == False and topX.nnz > maxprint:
        half = maxprint // 2
        out = tostr(A.row[:half], A.col[:half], A.data[:half])
        out += "\n  :\t:\n"
        half = maxprint - maxprint//2
        out += tostr(A.row[-half:], A.col[-half:], A.data[-half:])
    else:
        out = tostr(A.row, A.col, A.data)
    print(out)


def print_words_prob_distribution(m, index_names, word_ids=None, word_names=None, path=None):
    if word_ids is None and word_names is None:
        raise Exception("word_ids and word_names are all None")
    if word_ids is not None:
        features_name = tfidf_model.get_feature_names()
        word_names = list(map(lambda word_id: features_name[word_id], word_ids))
    if word_names is not None:
        word_ids = list(map(lambda word: tfidf_model.vocabulary_[word], word_names))

    # m can transform to ndarray, eg matrix.toarray()
    X2_df = pd.DataFrame(m.toarray())
    printed_words_df = X2_df[word_ids]
    printed_words_df.columns = word_names
    printed_words_df.index = index_names
    printed_words_df.rename_axis("category")

    print("-"*10 + "probability_distribution" + "-"*10)
    print(printed_words_df)

    if path is not None:
        printed_words_df.to_csv(path_or_buf=path, mode="a+")


def get_index_by_label(label_encoder_model, label):
    return label_encoder_model.classes_.tolist().index(label)

def get_label_by_index(label_encoder_model, index):
    return label_encoder_model.classes_[index]

def test():
    # coding=utf-8
    # document = ["本人 67年 离异 ， 1.6 高 ， 想 找 一个 相伴 到 老 的 老伴 。 夫妻 交友 。",
    #             "我 就 郁闷 了 ！ 夫妻 没有 信任 结婚 有意思 吗 ？"]
    document = ["I have a pen.",
                "I have an apple."]
    tfidf_model = TfidfVectorizer(norm="l2").fit(document)
    sparse_result = tfidf_model.transform(document)  # 得到tf-idf矩阵，稀疏矩阵表示法
    print(sparse_result)
    # (0, 3)	0.814802474667
    # (0, 2)	0.579738671538
    # (1, 2)	0.449436416524
    # (1, 1)	0.631667201738
    # (1, 0)	0.631667201738
    print(sparse_result.todense())  # 转化为更直观的一般矩阵
    # [[ 0.          0.          0.57973867  0.81480247]
    #  [ 0.6316672   0.6316672   0.44943642  0.        ]]
    print(tfidf_model.get_feature_names())
    # ['an', 'apple', 'have', 'pen']
    print(tfidf_model.vocabulary_)  # 词语与列的对应关系
    # {'have': 2, 'pen': 3, 'an': 0, 'apple': 1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--is_debug",
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="use debug mode")
    parser.add_argument(
        "--base_path",
        type=str,
        default="/Users/jiananliu/AISEALs/data/text_classification",
        help="refer to data_processor path tree")
    parser.add_argument(
        "--task_name",
        type=str,
        default="tribe_labels",
        help="task name")
    parser.add_argument(
        "--task_id",
        type=str,
        default="0",
        help="task id")
    parser.add_argument(
        "--num_top_k",
        type=int,
        default=100,
        help="the num of each category")

    FLAGS, unparsed = parser.parse_known_args()
    pp = pprint.PrettyPrinter().pprint
    pp("FLAGS: " + str(FLAGS))
    pp("unparsed: " + str(unparsed))
    print(FLAGS.is_debug)

    print("base_path:" + str(FLAGS.base_path))

    processor = get_processor(FLAGS.base_path, FLAGS.task_name, FLAGS.task_id, use_hdfs=False)

    docs_df = gen_documents(FLAGS.is_debug)

    df = docs_df['content']
    tfidf_model = get_tfidf_model(df)

    rs = transform_features(tfidf_model, df)
    print(rs)

    aa = rs.sum(axis=0).tolist()[0]
    word2tfidf = list(zip(tfidf_model.get_feature_names(), aa))

    word2tfidf.sort(key=lambda x: x[1], reverse=True)

    print(word2tfidf[: FLAGS.num_top_k])

    df = pd.DataFrame(word2tfidf[: FLAGS.num_top_k], columns=['word', 'weight'])

    df.to_csv(path_or_buf="top_total_{}_weights.csv".format(FLAGS.num_top_k), mode="w+", encoding='gbk')

    # for doc_df in docs_df.groupby('tribe_id'):
    #     tribe_id = doc_df[0]
    #     df = doc_df[1]['content']
    #     tfidf_model = get_tfidf_model(df)
    #
    #     rs = transform_features(tfidf_model, df)
    #     print(rs)
    #
    #     aa = rs.sum(axis=0).tolist()[0]
    #     word2tfidf = list(zip(tfidf_model.get_feature_names(), aa))
    #
    #     word2tfidf.sort(key=lambda x: x[1], reverse=True)
    #
    #     print(word2tfidf[: FLAGS.num_top_k])
    #
    #     df = pd.DataFrame(word2tfidf[: FLAGS.num_top_k], columns=['word', 'weight'])
    #
    #     df.to_csv(path_or_buf="top_{}_{}_weights.csv".format(tribe_id, FLAGS.num_top_k), mode="w+", encoding='gbk')

    # features_name = tfidf_model.get_feature_names()
    #
    # print('-' * 20)
    # print("top{} data:\n".format(k))
    # print_matrix_by_view(topX, processor.get_labels(), features_name, True)
    #
    # # ['怎么活', '哎', '被', '怎么办']
    # print_words_prob_distribution(X2, word_names=['累', '迷茫', '压力', '烦死', '前路', '难', '苦', '买房', '逼得', '辛苦', '迷失'], index_names=processor.get_labels())
    # for row in topX.tolil().rows:
    #     # word_names=["时候", "工作"]
    #     print_words_prob_distribution(X2, word_ids=row, index_names=processor.get_labels(), path="top_k_distribute.csv")
    #
    # def word_names2word_ids_func(str):
    #     word_names = str.strip().split(" ")
    #     word_name2word_id = lambda word: tfidf_model.vocabulary_[word] if word in tfidf_model.vocabulary_ else 0
    #     return list(map(word_name2word_id, word_names))
    #
    # docs_df['line'] = docs_df['line'].apply(word_names2word_ids_func)
    #
    # X2_df = pd.DataFrame(X2.toarray())
    # def transform_line_2_tfidf_feature(row):
    #     mask = X2_df.columns.map(lambda col: col in row['line'])
    #     return X2_df.loc[row['label']][mask]
    #
    # X2_by_cate_with_Nan = docs_df[['label', 'line']].apply(transform_line_2_tfidf_feature, axis=1)
    # X2_by_cate = X2_by_cate_with_Nan[topX.tocoo().col].fillna(0)
    #
    # # total_X = pd.concat([pd.DataFrame(X1.toarray()), X2_by_cate.reset_index(drop=True)],axis=1, ignore_index=True)
    # # print(total_X)

