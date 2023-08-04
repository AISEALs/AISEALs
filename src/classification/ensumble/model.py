from structs import AlgoId, OutputData
from multiple4LR.text_label import fasttext_model, svm_model, rf_predict, finally_label
from bert_client import bert_predict


class Model:
    schema = ['汽车价格', '工作', '职业规划/择业', '租房经历/经验', '工作介绍/前景', '工资/福利', '诈骗识别', '生活压力', '拼车', '八卦热点', '心情_情绪_想法表达', '征婚', '生活经历', '找老乡', '债务/贷款', '车辆评测', '求租/买房', '产品功能', '找对象', '感情/生活', '找工作', '招人/雇佣']

    # schema = ['汽车价格', '工作',  '租房经历/经验', '诈骗识别', '拼车', '八卦热点', '找老乡', '债务/贷款', '车辆评测', '求租/买房', '产品功能',  '感情/生活']

    def __init__(self, algo_id, model_func):
        self.algo_id = algo_id
        self.model_func = model_func

    def run(self, inputs):
        texts = [input.content for input in inputs]
        labels, results = self.model_func(texts)
        predicts = finally_label(labels, texts)

        outputs = []
        assert len(inputs) == len(results) == len(predicts)
        for input, logits, predict in zip(inputs, results, predicts):
            logits = ','.join([str(logits['__label__'+k]) for k in Model.schema])
            outputs.append(OutputData(self.algo_id, input.info_id, predict, logits))
        return outputs


class FaxttextModel(Model):
    def __init__(self):
        super(FaxttextModel, self).__init__(AlgoId.FASTTEXT, fasttext_model)


class SVMModel(Model):
    def __init__(self):
        super(SVMModel, self).__init__(AlgoId.SVM, svm_model)


class RFModel(Model):
    def __init__(self):
        super(RFModel, self).__init__(AlgoId.RF, rf_predict)


class BertModel(Model):
    def __init__(self, task_id):
        super(BertModel, self).__init__(AlgoId.BERT, bert_predict)
        self.task_id = task_id

    def run(self, inputs):
        outputs = self.model_func(inputs, self.task_id)
        return outputs

