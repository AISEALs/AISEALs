import os
import re
try:
    # pylint: disable=g-import-not-at-top
    import cPickle as pickle
except ImportError:
    # pylint: disable=g-import-not-at-top
    import pickle


def load_data_and_labels(dir, max_doc_length):
    # Load data from files
    files = os.listdir(dir)
    for fileName in files:
        if fileName.endswith(".classes"):
            continue
        filePath = os.path.join(dir, fileName)
        examples = list(open(filePath, "r",encoding='UTF-8').readlines())
        from data_processor.data_processor import Example
        examples = map(Example.deserialize_from_str, examples)
        examples = filter(lambda x: x!=None, examples)
        sp = lambda x: re.split("\x001| ", x.line)
        examples = map(lambda x: " ".join(sp(x)[0:max_doc_length]), examples)
        # examples = [" ".join(sp(x)[0:max_doc_length]) for x in examples]
        for line in examples:
            yield line


def gen_local_vocab(processor):
    max_document_length = 512
    min_frequency = 2
    # 序列长度填充或截取到512，删除词频<=2的词
    from common_tools import VocabularyProcessor
    vocab = VocabularyProcessor(max_document_length, min_frequency)

    x_text = load_data_and_labels(processor.cate_data_dir, max_document_length)
    vocab.fit(x_text)

    vocab_path = os.path.join(processor.task_dir, 'vocab.pickle')
    vocab.save(vocab_path)


def restore_vocab(processor, use_hdfs):
    vocab_path = os.path.join(processor.task_dir, 'vocab.pickle')
    # from common_tools import VocabularyProcessor
    # vocab = VocabularyProcessor.restore(vocab_path)
    if not use_hdfs:
        with open(vocab_path, 'rb') as f:
            return pickle.loads(f.read())
    else:
        from tools.tools import get_hdfs_client
        client = get_hdfs_client()
        with client.read(vocab_path) as f:
            return pickle.loads(f.read())


