#!/usr/bin/python
#encoding=utf-8

import logging
import re
import sys
import os

def write_file(sql,filename):
  try:
    file_obj = open(filename,'a+')
    file_obj.write(sql+'\n')
    file_obj.flush()
    file_obj.close()
  except:
    print ("write %s failed" % sql)


if __name__ == "__main__":
    data = open(sys.argv[1],'r')
    result = {}
    for line in data:
        line = line.replace("\r","").replace("\n","")
        label = line.split('\t')[0]
        if label not in result:
          result[label] = []
        content = line.split('\t')[1]
        result[label].append(content)

    for k,v in result.items():
      for i in range(0,len(v)):
        if i < 0.8 * len(v):
          write_file(k + '\t' + v[i],'train')
        else:
          write_file(k + '\t' + v[i],'test')
