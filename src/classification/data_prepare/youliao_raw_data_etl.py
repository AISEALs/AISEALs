import sys
import os
sys.path.append(os.getcwd())    # 在命令行中，把当前路径加入到sys.path中
import json
import traceback
import jieba
from common_tools import tokenization
from tools import tools
from data_processor.data_processor import Example
from common_tools.stop_word import load_stop_words


def convert_line2example(line, use_hdfs):
    stopwords = set()
    """Creates examples for the training and dev sets."""
    if len(stopwords) == 0:
        stopwords = load_stop_words('stop_words_ch_utf8.txt')
        print("load stop words size:{}".format(len(stopwords)))

    class YouliaoRecord(object):
        def __init__(self):
            self.title = None
            self.content = None
            self.label = None

    def _filter_stop_word(x):
        seg_text = jieba.cut(x.replace('\t',' ').replace('\n',' '))
        seg_text = filter(lambda y: y not in stopwords, seg_text)
        return seg_text

    def _deal_line(one_line):
        try:
            result = YouliaoRecord()
            if not 'zjb-' in one_line:
                return None
            one_line = one_line.split('zjb-')[1]
            decode_line = json.loads(one_line)
            title = decode_line['title']
            title = tools.clean_html(title)
            if not title:
                return None
            result.title = ' '.join(_filter_stop_word(title))

            content = decode_line['content']
            if not content:
                content = ""
            content = tools.clean_html(content)
            if not content:
                return None
            result.content = ' '.join(_filter_stop_word(content))

            category = decode_line['category']
            if not category:
                return None
            category = category.split('-')[0]
            result.label = category
            return result
        except Exception as e:
            print("abc" + str(e))
            traceback.print_exc()
            return None

    result = _deal_line(line)
    if not result:
        return None
    title = tokenization.convert_to_unicode(result.title)
    content = tokenization.convert_to_unicode(result.content)
    label = tokenization.convert_to_unicode(result.label)
    #spark分布式，后面会覆盖guid, cuid
    line = content + "\0001" + title
    return Example(guid=0, cuid=0, label=label, line=line)

