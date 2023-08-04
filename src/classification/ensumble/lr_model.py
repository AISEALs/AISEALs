import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.datasets import load_iris


def get_df(file_name):
    df = pd.read_csv(file_name, sep='\t', header=None, encoding='utf-8')
    df.columns = ["label", "features"]
    df = df.sample_from_session(frac=1)

    feature_df = df['features'].str.split(',', expand=True)
    label_df = df['label']

    # with open("features.csv") as csv_file:
    #     data_file = csv.reader(csv_file)
    #     n_samples = int(temp[0])
    #     n_features = int(temp[1])
    #     target_names = np.array(temp[2:])
    #     data = np.empty((n_samples, n_features))
    #     target = np.empty((n_samples,), dtype=np.int)
    #
    #     for i, ir in enumerate(data_file):
    #         data[i] = np.asarray(ir[:-1], dtype=np.float64)
    #         target[i] = np.asarray(ir[-1], dtype=np.int)
    #
    # return data, target, target_names
    return feature_df, label_df


def train_lr(train_x, train_y):
    logistic_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

    logistic_model.fit(train_x, train_y)
    return logistic_model


if __name__ == '__main__':
    train_x, train_y = get_df("train.csv")
    # train_x, train_y = df['feature'], df['label']
    logistic_model = train_lr(train_x, train_y)

    test_x, test_y = get_df("dev.csv")
    score = logistic_model.score(test_x, test_y)
    print("total mean accuracy is: {}".format(score))
