# -*- coding:utf-8 -*-
'''
@author:zjb
@file:youliao2dataset.py
测试生成tfrecord文件
把有料的原始文件生成tfrecord文件 *.train  , *.test ,vocabulary文件,label_index文件
'''

import argparse
import jieba
import json
from tools import tools
import tensorflow as tf
import random
import multiprocessing
import sys
from tensorflow.contrib import learn
import numpy as np


class YouliaoRecord(object):
    def __init__(self):
        self.title =""
        self.title_list = []
        self.content = ""
        self.content_list = []
        self.category = ""

inputdir ='data/youliao_raw_data/'
outputdir='data/youliao_dataset/'
stopfilepath = 'stop_words_ch_utf8.txt'
labelsmap = {}
#g_all_result={} #存放文件名->YouliaoRecord

stopwords=set()
stopwords.add(' ')

for l in open(stopfilepath, 'r',encoding='UTF-8').readlines():
    stopwords.add(l.strip())

def deal_json(line):
    try:
        sp=line.split('zjb-')
        #print(len(sp))
        if(len(sp)<2):
            return None

        line = sp[1]

        # text = line#.encode('utf-8').decode('utf-8')#.encode('utf-8')
        decode_line = json.loads(line)
        if 'videos' in decode_line:
            return None
        #content = decode_line['content']
        title = decode_line['title']
        title = tools.clean_html(title)
        title_list = jieba.cut(title.replace('\t',' ').replace('\n',' '))
        title_list = list(filter(lambda x:(x not in stopwords and len(x)>1),title_list))

        content = decode_line['content']
        content = tools.clean_html(content)
        content_list = jieba.cut(content.replace('\t',' ').replace('\n',' '))
        content_list = list(filter(lambda x:(x not in stopwords and len(x)>1) ,content_list))

        result = YouliaoRecord()
        result.title = ' '.join(title_list)
        result.content = ' '.join(content_list)
        category = decode_line['category']
        category = category.split('-')[0]
        result.category=category
        if labelsmap.get(category) is None:
            labelsmap[category]=1
        #result.y = labels.index(category)
        return result
    except Exception as e:
        #traceback.print_exc()
        return None

def deal_one_file(fullpath):
    res_list=[]
    print(fullpath+" start")
    sys.stdout.flush()
    cont=0
    with open(fullpath, 'r', encoding='UTF-8') as fr:
        for line in fr:
            result = deal_json(line)
            #cont=cont+1
            #if(cont>1000):######
            #    break
            if result != None:
                res_list.append(result)
    print(fullpath + " end "+str(len(res_list)))
    sys.stdout.flush()
    return res_list
    #g_all_result[fullpath]=res_list

def create_data(inputdir,outputdir,output_shards,eval_num,vocab_outfile,sequence_length):
    files = tf.gfile.ListDirectory(inputdir)
    fullfiles=list(map(lambda x:(inputdir+x),files))
    #print(" ".join(fullfiles))

    def _pick_output_shard():
        return random.randint(0, output_shards - 1)

    #for fileName in fullfiles:
    #    g_all_result[fileName] = []
    cores = multiprocessing.cpu_count()
    print("cores="+str(cores))
    #cores=2
    pool = multiprocessing.Pool(processes=cores)
    pool_list = []
    result_list=[]
    for ff in fullfiles:
        print(ff)
        pool_list.append(pool.apply_async(deal_one_file, (ff,)))  # 这里不能 get， 会阻塞进程
        #result_list.append(deal_one_file(ff))
    result_list = [x.get() for x in pool_list]
    pool.close()
    pool.join()
    allresult=[]
    for r in result_list:
        print(len(r))
        allresult = allresult+r#list(chain(*result_list))

    max_document_length = sequence_length
    min_frequency = 0
    # 序列长度填充或截取到100，删除词频<=2的词
    vocab = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency)
    #vocab = learn.preprocessing.VocabularyProcessor.restore("vocab.pickle")
    #vocab.restore("vocab.pickle")
    print("load vocab")

    wordlist = list(map(lambda x: (" ".join([x.title,x.content])), allresult))
    vocab.fit(wordlist)
    vocab.save(outputdir+vocab_outfile)
    title2id = np.array(list(vocab.transform(list(map(lambda x: x.title, allresult)))))
    #for ttt in range(100):
        #if title2id[ttt][0]==0 :
        #print(allresult[ttt].title + " cate:"+allresult[ttt].category)
        #print(title2id[ttt][0:10])
    #title2id = vocab.transform(map(lambda x: x.title, allresult))

    content2id = np.array(list(vocab.transform(map(lambda x: x.content, allresult))))
    #title_content2id = np.array(list(vocab.transform(map(lambda x: " ".join(x.title,x.content), allresult))))

    #tmp_content2id = [int(x) for x in content2id]
    tmpclass=list(set(list(map(lambda x:x.category,allresult))))
    classindex=map(lambda x: tmpclass.index(x.category), allresult)
    featureresult=[]
    for t_title,t_content,t_index in zip(title2id,content2id,classindex):
        features = {}
        features["title"] = tf.train.Feature(int64_list=tf.train.Int64List(value=t_title))
        features["content"] = tf.train.Feature(int64_list=tf.train.Int64List(value=t_content))
        features["category"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[t_index]))
        featureresult.append(features)
    random.shuffle(featureresult)
    writers = []
    test_writer = tf.python_io.TFRecordWriter("%s/%s.test" % (outputdir, "tf"))
    for r in featureresult[0:eval_num]:
        f = tf.train.Features(feature=r)
        example = tf.train.Example(features=f)
        test_writer.write(example.SerializeToString())
    for i in range(output_shards):
        writers.append(tf.python_io.TFRecordWriter("%s/%s-%05i-of-%05i.train" % (outputdir, "tf", i,output_shards)))
    for r in featureresult[eval_num:]:
        f = tf.train.Features(feature=r)
        example = tf.train.Example(features=f)

        writers[_pick_output_shard()].write(example.SerializeToString())

    for w in writers:
        w.flush()
        w.close()
    test_writer.flush()
    test_writer.close()
    with open(outputdir+"index.label",'w',encoding='UTF-8') as label_f:
        for l in tmpclass:
            label_f.write(l+" "+str(tmpclass.index(l))+"\n")

def main(argv):
  del argv
  create_data(FLAGS.input_dir,FLAGS.output_dir,FLAGS.output_shards,FLAGS.eval_num,FLAGS.vocab_outfile,FLAGS.sequence_length)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--input_dir",
      type=str,
      default="D:\\p_project\\data\\youliao\\raw_data/",
      help="Directory where the ndjson files are stored.")
  parser.add_argument(
      "--output_dir",
      type=str,
      default="D:\\p_project\\data\\youliao\\dataset/",
      help="Directory where to store the output TFRecord files.")
  parser.add_argument(
      "--output_shards",
      type=int,
      default=10,
      help="Number of shards for the train output.")
  parser.add_argument(
      "--eval_num",
      type=int,
      default=10,
      help="How many items load for evaluation.")
  parser.add_argument(
      "--vocab_outfile",
      type=str,
      default="vocab.v1",
      help="vocab")
  parser.add_argument(
      "--sequence_length",
      type=int,
      default=512,
      help="sequence_length.")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

