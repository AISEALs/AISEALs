import argparse
import pprint

import sys
import os
# 确保在命令行下执行，不出现ModuleNotFoundError: No module named ERROR
sys.path.append(os.getcwd())    # 在命令行中，把当前路径加入到sys.path中
sys.path.append(os.getcwd() + "/../")
from tribe_labels.analysize_text_frequent_words import gen_documents, gen_two_cate_documents, two_cate_documents_iter

# print("sys.path:{}".format("\n".join(sys.path)))
from data_processor.processor_manager import get_processor


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


def train_model(docs_df, model_name=""):
    docs_df['label'] = "__label__" + docs_df['label']

    import numpy as np
    msk = np.random.rand(len(docs_df)) < 0.8
    train_df = docs_df[msk]
    test_df = docs_df[~msk]
    print("train data num:{}".format(len(train_df)))
    print("test data num:{}".format(len(test_df)))

    # 构造train、test训练数据
    train_file_name = "data/train.csv"
    train_df[['label', 'line']].to_csv(train_file_name, index=False, sep="\t", header=False)
    test_file_name = "data/test.csv"
    test_df[['label', 'line']].to_csv(test_file_name, index=False, sep="\t", header=False)

    from fastText import train_supervised
    model = train_supervised(
        input=train_file_name, epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1
    )

    print("model_name: {}".format(model_name))
    print_results(*model.test(test_file_name))

    return model, test_df


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
        default="/Users/jiananliu/work/AISEALs/data/text_classification",
        help="refer to data_processor path tree")
    parser.add_argument(
        "--task_name",
        type=str,
        default="tribe_labels",
        help="task name")
    parser.add_argument(
        "--task_id",
        type=str,
        default="20190529",
        help="task id")
    parser.add_argument(
        "--multiple",
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help="是否是多分类问题")
    parser.add_argument(
        "--label_name",
        type=str,
        default="心情_情绪_想法表达",
        help="refer to data_processor path tree")

    FLAGS, unparsed = parser.parse_known_args()
    pp = pprint.PrettyPrinter().pprint
    pp("FLAGS: " + str(FLAGS))
    pp("unparsed: " + str(unparsed))
    print(FLAGS.is_debug)

    print("base_path:" + str(FLAGS.base_path))

    # labels path: FLAGS.base_path + FLAGS.task_name + FLAGS.task_id
    processor = get_processor(FLAGS.base_path, FLAGS.task_name, FLAGS.task_id)

    if FLAGS.multiple:
        df = gen_documents(processor, FLAGS.is_debug)
        version = "multi_1.0"
        model, test_df = train_model(df, version)
        model_path = "models/fasttext/fasttext_model_{}.bin".format(version)
        print("-------------save models:{}---------------".format(model_path))
        model.save_model(model_path)
    else:
        docs_df = gen_two_cate_documents(processor, FLAGS.is_debug, FLAGS.label_name, "其他")
        model, test_df = train_model(docs_df, FLAGS.label_name)
        model_path = "models/fasttext/fasttext_model_{}.bin".format(FLAGS.label_name)
        print("-------------save models:{}---------------".format(model_path))
        model.save_model(model_path)
        # for cate_name, df in two_cate_documents_iter(processor, FLAGS.is_debug, "其他"):
        #     models = train_model(df, cate_name)
        #     model_path = "models/fasttext/fasttext_model_{}.bin".format(cate_name)
        #     print("-------------save models:{}---------------".format(model_path))
        #     models.save_model(model_path)
            # break
        real_labels = test_df.label
        results = model.precision(test_df.line.tolist())
        test_df['predict'] = results[0]
        test_df['predict'] = test_df['predict'].map(lambda x: x[0])
        test_df['prob'] = results[1]
        df = test_df[['label', 'predict', 'prob', 'line', 'label_copy']]

        error_df = df[df['label'] != df['predict']]

        error_df_2 = error_df[error_df['prob'] > 0.9]
        error_file = "data/error_df.csv"
        error_df_2[['label_copy', 'predict', 'prob', 'line']].to_csv(error_file, index=False, sep="\t")
