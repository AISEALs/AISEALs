import argparse
import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# import scipy.sparse as sp
import pickle

import sys
import os
# 确保在命令行下执行，不出现ModuleNotFoundError: No module named ERROR
sys.path.append(os.getcwd())    # 在命令行中，把当前路径加入到sys.path中
# sys.path.append("/Users/jiananliu/work/python/AISEALs/text_classification")
print("sys.path:{}".format("\n".join(sys.path)))
from data_processor.processor_manager import get_processor
from common_tools.maxtrix_trans.select_top_k import top_k
from common_tools import stop_word

max_num = 200

def compose2document(processor, label, is_debug, num=None):
    one_cate_file_info = processor.get_file_handler(label)
    df = one_cate_file_info.get_df()
    if is_debug:
        df = df.sample_from_session(10)
    if num == None:
        return df
    if len(df) >= num:
        df = df.sample_from_session(num)
    return df

def gen_documents(processor, is_debug):
    docs_df = pd.concat([compose2document(processor, label, is_debug) for label in processor.get_labels()]).reset_index(drop=True)
    return docs_df.sample_from_session(frac=1) #打乱顺序
    # return docs_df.reset_index()

def gen_two_cate_documents(processor, is_debug, cate_name, other_name):
    if cate_name not in processor.get_labels():
        raise "cate_name:{} not in labels".format(cate_name)

    cate_df = compose2document(processor, cate_name, is_debug, num=100000)

    other_doc_df = pd.concat([compose2document(processor, label, is_debug, num=10000) for label in processor.get_labels() if label != cate_name]).reset_index(drop=True)

    # 保留之前的类别，后面DEBUG
    cate_df['label_copy'] = cate_df['label']
    other_doc_df['label_copy'] = other_doc_df['label']

    msk = other_doc_df['label'] == cate_name
    other_doc_df['label'].where(cond=msk, other=other_name, inplace=True)

    frac = len(cate_df) / len(other_doc_df)
    df = pd.concat([cate_df, other_doc_df.sample_from_session(frac=frac)])

    return df.sample_from_session(frac=1) #打乱顺序


def two_cate_documents_iter(processor, is_debug, other_name):
    doc_df = pd.concat([compose2document(processor, label, is_debug) for label in processor.get_labels()]).reset_index(drop=True)

    for cate_name in processor.get_labels():
        msk = doc_df['label'] == cate_name
        cate_df = doc_df[msk]
        other_doc_df = doc_df[~msk]
        if len(cate_df) < max_num:#10000:
            continue
        other_doc_df['label'].where(cond=msk, other=other_name, inplace=True)

        frac = len(cate_df) / len(other_doc_df)
        df = pd.concat([cate_df, other_doc_df.sample_from_session(frac=frac)])
        print("cate_name: {} lines num: {} total lines num: {}".format(cate_name, len(cate_df), len(df)))

        yield cate_name, df.sample_from_session(frac=1) #打乱顺序


def gen_cate_documents(docs_df):
    cate2doc_ser = docs_df[['label', 'line']].groupby('label').apply(lambda df: ' '.join(df['line']))
    # pd.DataFrame({'Gene':s.index, 'count':s.values}), 两种方法
    cate2doc_df = pd.DataFrame(cate2doc_ser).reset_index()
    cate2doc_df.columns = ['label', 'line']
    return cate2doc_df

def get_tfidf_model(docs_df, working_path):
    stop_file_path = os.path.join(working_path, 'files/stop_words_tribe_labels.txt')
    stop_words = stop_word.load_stop_words(stop_file_path)
    tfidf_model = TfidfVectorizer(stop_words=stop_words, token_pattern=r"(?u)\b\w+\b").fit(docs_df["line"])
    print("vocabulary size: {}".format(len(tfidf_model.vocabulary_)))
    return tfidf_model


def get_label_encoder_model(docs_df):
    label_encoder = LabelEncoder().fit(docs_df["label"])
    print("labels: {}".format(label_encoder.classes_))
    return label_encoder


def transform_features(tfidf_model, label_encoder, x, y):
    x_ = tfidf_model.transform(x)  # 得到tf-idf矩阵，稀疏矩阵表示法
    print("vocab size: {}".format(len(tfidf_model.vocabulary_)))
    print("X matrix shape: {}".format(x_.shape))

    y_ = label_encoder.transform(y)

    return x_, y_


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

    docs_df = gen_documents(processor, FLAGS.is_debug)

    tfidf_model = get_tfidf_model(docs_df, FLAGS.working_path)

    label_encoder_model = get_label_encoder_model(docs_df)

    cate2doc_df = gen_cate_documents(docs_df)

    # X2 shape [9x38411] sparse matrix
    X2, Y2 = transform_features(tfidf_model, label_encoder_model, cate2doc_df['line'], cate2doc_df['label'])

    k = FLAGS.num_top_k
    # sparse matrix shape:[9*vocab_size] size: 9*k
    topX = top_k(X2, k)

    features_name = tfidf_model.get_feature_names()

    print('-' * 20)
    print("top{} data:\n".format(k))
    print_matrix_by_view(topX, processor.get_labels(), features_name, True)

    # ['怎么活', '哎', '被', '怎么办']
    print_words_prob_distribution(X2, word_names=['累', '迷茫', '压力', '烦死', '前路', '难', '苦', '买房', '逼得', '辛苦', '迷失'], index_names=processor.get_labels())
    for row in topX.tolil().rows:
        # word_names=["时候", "工作"]
        print_words_prob_distribution(X2, word_ids=row, index_names=processor.get_labels(), path="top_k_distribute.csv")

    def word_names2word_ids_func(str):
        word_names = str.strip().split(" ")
        word_name2word_id = lambda word: tfidf_model.vocabulary_[word] if word in tfidf_model.vocabulary_ else 0
        return list(map(word_name2word_id, word_names))

    docs_df['line'] = docs_df['line'].apply(word_names2word_ids_func)
    docs_df['label'] = label_encoder_model.transform(docs_df['label'].values)

    X2_df = pd.DataFrame(X2.toarray())
    def transform_line_2_tfidf_feature(row):
        mask = X2_df.columns.map(lambda col: col in row['line'])
        return X2_df.loc[row['label']][mask]

    X2_by_cate_with_Nan = docs_df[['label', 'line']].apply(transform_line_2_tfidf_feature, axis=1)
    X2_by_cate = X2_by_cate_with_Nan[topX.tocoo().col].fillna(0)

    # total_X = pd.concat([pd.DataFrame(X1.toarray()), X2_by_cate.reset_index(drop=True)],axis=1, ignore_index=True)
    # print(total_X)

