#!/usr/bin/python
#encoding=utf-8

from life_predict import *
from human_rule import *
from fastText import load_model
from img_label import *

classifier1 = load_model('model/12cate.bin')
classifier2 = load_model('model/life_emotion_5_cate.bin')
classifier3 = load_model('model/5_cate_job.bin')

label2id = {}
with open("label_ids.txt", "r") as f:
    for line in f:
        line = line.replace('\n','').split('\t')
        label2id[line[1]] = str(line[0])


def predict_label(infos):
  start_1 = time.time()
  today = time.strftime("%Y%m%d",time.localtime(time.time()))
  texts = []
  for i in infos:
      texts.append(i[2])

  img_predict_result,success = img_predict(infos,today)
  start_2 = time.time()
  write_file("img predict time:%s" % (start_2 - start_1),"log/timer.%s" % today)
  if not success:
      write_file("img predict fail,return empty","log/log.wf")
      return []

  result = []

  labels_1, probabs_1 = classifier1.precision(texts)
  labels_2, probabs_2 = classifier2.precision(texts)
  labels_3, probabs_3 = classifier3.precision(texts)

  start_3 = time.time()
  write_file("text classifier time:%s" % (start_3 - start_2),"log/timer.%s" % today)
  for i in range(0,len(labels_1)):
      if labels_1[i][0] == "__label__感情/生活":
          if labels_2[i][0] == "__label__生活压力" or labels_2[i][0] == "__label__生活经历":
              life_label = life_predict_labels(' '.join(texts[i]))[0]
              by_rule_label = get_by_rule(texts[i],life_label)
          else:
              by_rule_label = get_by_rule(texts[i],labels_2[i][0])

          label_name =   by_rule_label.replace("__label__","").replace("_","/")
          if label_name not in label2id:
              write_file("strange:%s" % by_rule_label,"log/log.wf")
              result.append([['874'],float(probabs_1[i][0]),float(probabs_1[i][0]) *      float(probabs_2[i][0]),label_name])
          else:
              result.append([[label2id[label_name]],float(probabs_1[i][0]),float(probabs_1[i][0]) * float(probabs_2[i][0]),label_name])

      elif labels_1[i][0] == "__label__工作":
          by_rule_label = get_by_rule(texts[i],labels_3[i][0])
          label_name =   by_rule_label.replace("__label__","").replace("_","/")
          if label_name not in label2id:
            write_file("strange:%s" % by_rule_label,"log/log.wf")
            result.append([['816'],float(probabs_1[i][0]),float(probabs_1[i][0]) * float(probabs_3[i][0]),label_name])
          else:
            result.append([[label2id[label_name]],float(probabs_1[i][0]),float(probabs_1[i][0]) *float(probabs_3[i][0]),label_name])


      else:
          label_name = labels_1[i][0].replace("__label__","").replace("_","/")
          if label_name not in label2id:
            write_file("strange:%s" % labels_1[i][0],"log/log.wf")
            result.append([['874'],float(probabs_1[i][0]),float(probabs_1[i][0]),label_name])
          else:
              if int(label2id[label_name]) == 898:
                  result.append([['874'],float(probabs_1[i][0]),float(probabs_1[i][0]),label_name])
                  write_file(str(infos[i][0]),'log/watch.%s' % today)
              else:
                  result.append([[label2id[label_name]],float(probabs_1[i][0]),float(probabs_1[i][0]),label_name])

  for i in range(0,len(result)):
      if infos[i][0] in img_predict_result:
          result[i] = img_predict_result[infos[i][0]]
      write_file(str(infos[i][0]) + '\t' + ",".join(result[i][0]) + '\t' +  result[i][3] + '\t' + labels_1[i][0] + '\t' + str(probabs_1[i][0]) + '\t' + labels_2[i][0] + '\t' + str(probabs_2[i][0]) + '\t' + labels_3[i][0] + '\t' + str(probabs_3[i][0]) + '\t' + ' '.join(texts[i]),'log/predict.%s' % today)

  start_4 = time.time()
  write_file("text process time:%s" % (start_4 - start_3),"log/timer.%s" % today)
  return result


if __name__ == "__main__":
    infos = [(1,[],'只要 关系 好 相处 融洽 就 行'),(2,['https://pic8.58cdn.com.cn/mobile/big/n_v2ffe915136448467284a3b38ce073d48d.jpg'],'刚 开始 特别 讨厌 厨师 这个 行业 。 很 排斥 为什么 下班 这么晚 ！ 错过 了 约 朋友 的 时间 ！ 逢年过节 不能 休息 ！ 现在 看见 家人 和 朋友 吃 我 做 的 饭 ， 没 以前 排斥 的 那么 厉害 了 ！ 也 习惯 了 一个 人 挤 地铁 ， 两点 一线 的 生活 ！ 虽然 很 乏味 ， 总是 会 有 那么 一点 快乐 让 我 反复 的 回味 ！'),(3,['https://pic8.58cdn.com.cn/mobile/big/n_v2ffe912717166418e9cbd69c86c0ea3a4.jpg','https://pic8.58cdn.com.cn/mobile/big/n_v2ffe960a72b214f73b1c648cb0638d2c9.jpg'],'总会 有人 明白 你 的 坚强 ， 即使 你 给 自己 裹 了 刺 ， 他 也 会 笑 着 去 拥抱 你 。 早安 ！')]
    result = predict_label(infos)
    for i in range(0,len(result)):
        print(i,result[i][0],result[i][1])



