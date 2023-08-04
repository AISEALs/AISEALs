#from transformers import AdamW
import logging
from torch.optim import AdamW
import sys
sys.path.append('..')

def build_optim(model, args):
    # model_lr = {'others': config.others_lr, 'roberta': config.bert_lr}
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = []
    # optimizer_grouped_parameter_names = []
    # for layer_name in model_lr:
    #     lr = model_lr[layer_name]
    #     if layer_name != 'others':  # 设定特定lr的layer
    #          optimizer_grouped_parameters += [
    #             {
    #                 "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
    #                                                                       and layer_name in n)],
    #                 "weight_decay": config.weight_decay,
    #                 "lr": lr,
    #             },
    #             {
    #                 "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
    #                                                                       and layer_name in n)],
    #                 "weight_decay": 0.0,
    #                 "lr": lr,
    #             },
    #          ]
    #          optimizer_grouped_parameter_names += [
    #              {
    #                  "params": [n for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
    #                                                                        and layer_name in n)],
    #                  "weight_decay": config.weight_decay,
    #                  "lr": lr,
    #              },
    #              {
    #                  "params": [n for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
    #                                                                        and layer_name in n)],
    #                  "weight_decay": 0.0,
    #                  "lr": lr,
    #              },
    #          ]
    #     else:  # 其他，默认学习率
    #         optimizer_grouped_parameters += [
    #             {
    #                 "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
    #                                                                       and not any(name in n for name in model_lr))],
    #                 "weight_decay": config.weight_decay,
    #                 "lr": lr,
    #             },
    #             {
    #                 "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
    #                                                                       and not any(name in n for name in model_lr))],
    #                 "weight_decay": 0.0,
    #                 "lr": lr,
    #             },
    #         ]
    #         optimizer_grouped_parameter_names += [
    #             {
    #                 "params": [n for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
    #                                                                       and not any(name in n for name in model_lr))],
    #                 "weight_decay": config.weight_decay,
    #                 "lr": lr,
    #             },
    #             {
    #                 "params": [n for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
    #                                                                       and not any(name in n for name in model_lr))],
    #                 "weight_decay": 0.0,
    #                 "lr": lr,
    #             },
    #         ]
    #
    # optimizer = AdamW(
    #     optimizer_grouped_parameters,
    #     lr=model_lr['roberta'],
    #     eps=config.adam_epsilon
    # )
    # logging.info(optimizer_grouped_parameter_names)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer
