from enum import Enum


class InputData():
    def __init__(self, info_id, label, content):
        self.info_id = info_id
        self.label = label
        self.content = content


class OutputData():
    def __init__(self, algo_id, info_id, predict, logits):
        self.algo_id = algo_id
        self.info_id = info_id
        self.predict = predict
        self.logits = logits        # list，概率列表

    def __str__(self):
        return ','.join([str(self.algo_id.value), str(self.info_id), str(self.predict), str(self.logits)])

    def features(self):
        return self.logits


class AlgoId(Enum):
    BERT = 1
    FASTTEXT = 2
    SVM = 3
    RF = 4
