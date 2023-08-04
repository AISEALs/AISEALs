#!/usr/bin/python
#encoding=utf-8

import re
import fastText
import sys
import jieba
import os

zhenghun = ["结婚","过日子","征婚","余生","老婆","共度","一生","另一半","媳妇","为伴","一辈子","下半生","后半生","半生","辈子","夫妻","度过","半辈子","伴侣","老公","终生","终身","嫁给","嫁到","嫁","娶","婚后","女婿","成婚","寻偶","配偶","白头"]
object = ["我","老公","老婆","对象","父母","孩子","爸妈","爸","妈","同学","同事","朋友","男朋友","女朋友","男友","女友","自己","媳妇","儿子","女儿","女婿","丈夫","妻子"]

PATTERN = re.compile(r'''换(.{0,5})工作''',re.I)
zeye = ["不知道","不晓得","不知","不清楚","做什么","干什么","迷茫","不想"]


def get_by_rule(text,label):
  if label == "找对象" :
    if len(list(set(text).intersection(set(zhenghun)))) > 0:
      return "征婚"
    else:
      return "找对象"
  elif label == "生活经历":
    if len(list(set(text).intersection(set(object)))) > 0:
      return "生活经历"
    else:
      return "心情/情绪/想法表达"
  elif label == "找工作":
      if len(list(set(text).intersection(set(zeye)))) > 0:
          return "职业规划/择业"
      else:
          if len(PATTERN.findall(''.join(text).replace(' ',''))) > 0:
              return "职业规划/择业"
          for i in zeye:
              if ''.join(text).replace(' ','').find(i) != -1:
                  return "职业规划/择业"
          else:
              return "找工作"
  else:
      return label






