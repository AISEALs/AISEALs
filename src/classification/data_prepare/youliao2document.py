# -*- coding:utf-8 -*-
'''
@author:zjb
@time:2018/9/1812:22
输入是日期格式的原始文件，按照category作为目录，每个文件中，每行都是纯json

'''

import json
import traceback
import os
import sys


sourcefile='2018-09-13'
basedir ='D:\\tmp\\youliao\\youliao_input'
outdir ='D:\\tmp\\youliao\\youliao_output'
outfiles={}
def deal(line):
    try:
        line = line.split('zjb-')[1]

        # text = line#.encode('utf-8').decode('utf-8')#.encode('utf-8')
        decode_line = json.loads(line)
        #content = decode_line['content']
        #title = decode_line['title']
        #x = title + '' + content
        # print(content)
        # print('-' * 20)
        # print(content2)
        # print('-' * 20)

        category = decode_line['category']
        #category = category.split('-')[0]

        return category , line
    except Exception as e:
        #traceback.print_exc()
        return None,None


def write_cate_file(cate, line):
    sp =cate.split('-')
    ttt= "/".join(sp)
    tmpdir=outdir+"/"+"/".join(sp)+"/"
    tmpfile = tmpdir+sourcefile
    f=outfiles.get(tmpfile)
    if f==None :
        if not os.path.exists(tmpdir) :
            os.makedirs(tmpdir)
        writer = open(tmpfile, 'w', encoding='UTF-8')
        outfiles[tmpfile] = writer
    else :
        #print("ssssss")
        outfiles[tmpfile].write(line)



def flush_files(map):
    for f in map.values():
        if type(f) == "Dict" :
            flush_files(f)
        else:
            f.flush()
            f.close()

def create_data():
    catenum = 0
    catenum += 1
    # files = os.listdir(basedir)
    # files = ['train_data.txt']
    count = 0
    try:
        with open(basedir+'/'+sourcefile,'r',encoding='UTF-8') as fr:
            for line in fr:
                cate,line = deal(line)
                if line != None:
                    write_cate_file(cate,line)
        count += 1
    except Exception as e:
        traceback.print_exc()


    flush_files(outfiles)


if __name__ == '__main__':
    if(len(sys.argv)>1) :
        sourcefile = sys.argv[1]
    create_data()
