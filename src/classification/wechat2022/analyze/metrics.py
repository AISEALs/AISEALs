import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import sys
import analyze_feature

sys.path.append('..')
from category_id_map import lv2id_to_lv1id

results = []
with open('./data/result.txt') as f:
    i = 0
    for line in f:
        results.append(line)

line = results[-1]
sp = line.split(", ")
y_pred = list(map(int, sp[0].split(":")[1].strip().split(",")))
y_true = list(map(int, sp[1].split(":")[1].strip().split(",")))
i += 1
print(f"epoch: {i}")

lv1_preds = [lv2id_to_lv1id(lv2id) for lv2id in y_pred]
lv1_labels = [lv2id_to_lv1id(lv2id) for lv2id in y_true]

lv2_f1_micro = f1_score(y_true, y_pred, average='micro')
lv2_f1_macro = f1_score(y_true, y_pred, average='macro')
lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

eval_results = {'lv1_f1_micro': lv1_f1_micro,
                'lv1_f1_macro': lv1_f1_macro,
                'lv2_f1_micro': lv2_f1_micro,
                'lv2_f1_macro': lv2_f1_macro,
                'mean_f1': mean_f1}
print(eval_results)
labels = list(set(y_true))
print('labels', labels)
probs = f1_score(y_true, y_pred, labels=labels, average=None)
df = pd.DataFrame(list(zip(labels, probs)))
df.columns = ['lv2id', 'f1_score']

df2 = analyze_feature.get_labels_df()
cate2num = df2.groupby('lv2id').size().reset_index()
cate2num.columns = ['lv2id', 'num']

df3 = df.join(cate2num.set_index('lv2id'), on='lv2id')
df3.sort_values(by='f1_score')


