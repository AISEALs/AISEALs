# -*- coding:utf-8 -*-
"""
@author:Levy
@file:dealtext.py
@time:2018/9/1812:22
"""
import jieba
import os
import codecs

basedir ="D:\\doc\\语料库\\Reduced2\\Reduced2\\Reduced\\"
stopfilepath = "D:\\doc\\语料库\\stop_words_ch_utf8.txt"
dir_list =["C000008","C000010","C000013","C000014","C000016","C000020","C000022","C000023","C000024"]

stopwords=set()
stopwords.add(" ")

for l in open(stopfilepath, "r",encoding='UTF-8').readlines():
    stopwords.add(l)

catenum = 0
for d in dir_list:
    ftrain = open("./data/sougoutrain/"+d+".txt","w",encoding='UTF-8')
    catenum += 1
    indir = basedir + d + '/'
    files = os.listdir(indir)
    count = 0
    for fileName in files:
        try:
            filepath = indir + fileName
            with open(filepath,'r',encoding='UTF-8') as fr:
                text = fr.read()
            #print(type(text))
            if text[:3] == codecs.BOM_UTF8:
                text = text[3:]
            text = text#.encode("utf-8").decode("utf-8")#.encode("utf-8")
            seg_text = jieba.cut(text.replace("\t"," ").replace("\n"," "))
            seg_text = filter(lambda x:x not in stopwords ,seg_text)
            outline = " ".join(seg_text)
            count += 1
        except Exception as e:
            print(e)
            print(d+"->"+fileName+" :err")
            continue
        #print(outline)
        #outline = outline.encode("utf-8") + "\t__label__" + e + "\n"
#         print outline
#         break
        ftrain.write(outline+"\n")

        #print(count)
        if(count%100==0):

            ftrain.flush()
            print(d+"->"+str(count))

    ftrain.close()
