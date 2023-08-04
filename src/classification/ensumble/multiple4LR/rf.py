#encoding=utf-8
import os   #用于读取文件
import pandas
import numpy
import codecs
import sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn import datasets
import pickle

def constructDataset(path):
    f = codecs.open(path, 'r', encoding='utf-8')
    content = f.read()
    f.close()
    label_list = []
    corpus_list = []
    for line in content.split('\n'):
        if len(line.split('\t')) != 2:
            continue
        label = line.replace('\n', '').split('\t')[0].replace("__label__","")
        # if label not in labels:
        #     continue
        label_list.append(line.replace('\n', '').split('\t')[0])
        corpus_list.append(line.replace('\n', '').split('\t')[1])
    return label_list, corpus_list

cur_dirname = os.path.dirname(os.path.abspath(__file__))

model1 = os.path.join(cur_dirname, "./randomforest/12_train.pkl")
vec1  = os.path.join(cur_dirname, "./randomforest/feature_12_train.pkl")


model2 = os.path.join(cur_dirname, "./randomforest/life_emotion_train_2.pkl")
vec2 = os.path.join(cur_dirname, "./randomforest/feature_life_emotion_train_2.pkl")

model3 = os.path.join(cur_dirname, "./randomforest/5_cate_job_train.pkl")
vec3 = os.path.join(cur_dirname, "./randomforest/feature_5_cate_job_train.pkl")

rf_model1 = joblib.load(model1)
loaded_vec1 = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(vec1, "rb")))

rf_model2 = joblib.load(model2)
loaded_vec2 = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(vec2, "rb")))

rf_model3 = joblib.load(model3)
loaded_vec3 = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(vec3, "rb")))

def rf_predict(texts):
    test_vector1 = loaded_vec1.transform(texts)
    tfidf1 = TfidfTransformer(use_idf=False).fit_transform(test_vector1)

    test_vector2 = loaded_vec2.transform(texts)
    tfidf2 = TfidfTransformer(use_idf=False).fit_transform(test_vector2)

    test_vector3 = loaded_vec3.transform(texts)
    tfidf3 = TfidfTransformer(use_idf=False).fit_transform(test_vector3)


    probs = []
    first_label = []


    predict_prob1 = rf_model1.predict_proba(tfidf1)
    predict_label1 = rf_model1.precision(tfidf1)

    predict_prob2 = rf_model2.predict_proba(tfidf2)
    predict_label2 = rf_model2.precision(tfidf2)

    predict_prob3 = rf_model3.predict_proba(tfidf3)
    predict_label3 = rf_model3.precision(tfidf3)

    for i in range(0,len(predict_label1)):
        result = {}
        for j in range(0,len(rf_model1.classes_)):
            result[rf_model1.classes_[j]] = predict_prob1[i][j]
        for j in range(0,len(rf_model2.classes_)):
            result[rf_model2.classes_[j]] = predict_prob2[i][j]
        for j in range(0,len(rf_model3.classes_)):
            result[rf_model3.classes_[j]] = predict_prob3[i][j]

        #print(predict_label1[i],predict_label2[i],predict_label3[i])
        first_label.append([predict_label1[i],predict_label2[i],predict_label3[i]])
        probs.append(result)


    return first_label,probs



if __name__ == "__main__":
    infos = ['只要 关系 好 相处 融洽 就 行','刚 开始 特别 讨厌 厨师 这个 行业 。 很 排斥 为什么 下班 这么晚 ！ 错过 了 约 朋友 的 时间 ！ 逢年过节 不能 休息 ！ 现在 看见 家人 和 朋友 吃 我 做 的 饭 ， 没 以前 排斥 的 那么 厉害 了 ！ 也 习惯 了 一个 人 挤 地铁 ， 两点 一线 的 生活 ！     虽然 很 乏味 ， 总是 会 有 那么 一点 快乐 让 我 反复 的 回味 ！','总会 有人 明白 你 的 坚强 ，  即使 你 给 自己 裹 了 刺 ， 他 也 会 笑 着 去 拥抱 你 。 早安 ！']
    result = rf_predict(infos)
    for i in result:
        for k,v in i.items():
            print(k,v)




