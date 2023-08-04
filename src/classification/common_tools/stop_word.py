def load_stop_words(file_name):
    stopwords = set()
    stopwords.add(' ')
    stopFileName = file_name

    with open(stopFileName, 'r', encoding='UTF-8') as f:
        for word in f:
            stopwords.add(word.strip())

    return stopwords

