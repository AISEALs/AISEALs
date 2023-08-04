#!/usr/bin/python
#encoding=utf-8

from online.human_rule import *
from tgrocery import Grocery
from .rf import *
from online.life_predict import *
import fastText
from conf import config

l2cate_model_path = os.path.join(config.working_dirname, "online/model/12cate_2.bin")
life_emotion_path = os.path.join(config.working_dirname, "online/model/life_emotion_5_cate_2.bin")
cate5_job_path = os.path.join(config.working_dirname, "online/model/5_cate_job.bin")

classifier1 = fastText.load_model(l2cate_model_path)
classifier2 = fastText.load_model(life_emotion_path)
classifier3 = fastText.load_model(cate5_job_path)

cur_dirname = os.path.dirname(os.path.abspath(__file__))
svm1 = Grocery(os.path.join(cur_dirname, './grocery/12_train'))
svm1.load()
svm2 = Grocery(os.path.join(cur_dirname, './grocery/life_emotion_train_2'))
svm2.load()
svm3 = Grocery(os.path.join(cur_dirname, './grocery/5_cate_job_train'))
svm3.load()

def fasttext_model(texts):
    result = []
    first_label = []

    labels_1, probabs_1 = classifier1.precision(texts, 100)
    labels_2, probabs_2 = classifier2.precision(texts, 100)
    labels_3, probabs_3 = classifier3.precision(texts, 100)

    for i in range(0,len(labels_1)):
        tmp = {}
        for j in range(0,len(labels_1[i])):
            tmp[labels_1[i][j]] = probabs_1[i][j]
        for j in range(0,len(labels_2[i])):
            tmp[labels_2[i][j]] = probabs_2[i][j]
        for j in range(0,len(labels_3[i])):
            tmp[labels_3[i][j]] = probabs_3[i][j]

        result.append(tmp)


        first_label.append([labels_1[i][0],labels_2[i][0],labels_3[i][0]])

    return first_label,result


def svm_model(texts):
    result = []
    first_label = []
    for text in texts:
        result1 = svm1.precision(text)
        result2 = svm2.precision(text)
        result3 = svm3.precision(text)
        result.append({**(result1.dec_values), **(result2.dec_values), **(result3.dec_values)})
        first_label.append([str(result1),str(result2),str(result3)])

    return first_label,result

def finally_label(labels,texts):
    finally_labels = []
    for i in range(0,len(texts)):
        if labels[i][0] == "__label__感情/生活":
            if labels[i][1] == "__label__生活压力" or labels[i][1] == "__label__生活经历":
                life_label = life_predict_labels(' '.join(texts[i]))[0]
                by_rule_label = get_by_rule(texts[i],life_label)
            else:
                by_rule_label = get_by_rule(texts[i],labels[i][1])
        elif labels[i][0] == "__label__工作":
            by_rule_label = get_by_rule(texts[i],labels[i][2])
        else:
            by_rule_label = labels[i][0]

        if by_rule_label.find("__label__") == -1:
            by_rule_label = "__label__" + by_rule_label
        #print("by_rule_label:%s" % by_rule_label)
        finally_labels.append(by_rule_label)
    return finally_labels

def predict_label(texts):

  ft_label,ft_result = fasttext_model(texts)
  svm_label,svm_result = svm_model(texts)
  rf_label,rf_result = rf_predict(texts)

  '''for i in range(0,3):
      print(ft_label[i],ft_result[i])
      print(svm_label[i],svm_result[i])
      print(rf_label[i],rf_result[i])'''


  ft_feature = finally_label(ft_label,texts)
  svm_feature = finally_label(svm_label,texts)
  rf_feature = finally_label(rf_label,texts)
  for i in range(0,len(texts)):
      ft_result[i]['finally_label'] = ft_feature[i]
      svm_result[i]['finally_label'] = svm_feature[i]
      rf_result[i]['finally_label'] = rf_feature[i]

  return ft_result,svm_result,rf_result


if __name__ == "__main__":
    texts = ['只要 关系 好 相处 融洽 就 行','刚 开始 特别 讨厌 厨师 这个 行业 。 很 排斥 为什么 下班 这么晚 ！ 错过 了 约 朋友 的 时间 ！ 逢年过节 不能 休息 ！ 现在 看见 家人 和 朋友 吃 我 做 的 饭 ， 没 以前 排斥 的 那么 厉害 了 ！ 也 习惯 了 一个 人 挤 地铁 ， 两点 一线 的 生活 ！ 虽然 很 乏味 ， 总是 会 有 那么 一点 快乐 让 我 反复 的 回味 ！','总会 有人 明白 你 的 坚强 ， 即使 你 给 自己 裹 了 刺 ， 他 也 会 笑 着 去 拥抱 你 。 早安 ！']
    r1,r2,r3 = predict_label(texts)
    print('\n\n\n')
    for i in range(0,len(r1)):
        for k,v in r1[i].items():
            print(k,v)
        for k,v in r2[i].items():
            print(k,v)
        for k,v in r3[i].items():
            print(k,v)


