import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot(data, title):
    sns.set_style('dark')
    f, ax = plt.subplots()
    ax.set(ylabel='frequency')
    ax.set(xlabel='height(blue) / weight(green)')
    ax.set(title=title)
    sns.distplot(data, color='blue')
    plt.show()

df = pd.read_csv('../data/DPP_RescaleScore.log', sep='|', header=None)

raw_score_list = []
z_score_list = []
norm_score_list = []
scores_list = df[8].apply(lambda x: x.replace('scores:', '')).to_list()

for scores in scores_list:
    scores_sp = scores.split(',')
    for docid_scores in scores_sp:
        sp = docid_scores.split(':')
        if len(sp) == 2:
            second_part = sp[1].split('_')
            raw_score_list.append(float(second_part[0]))
            z_score_list.append(float(second_part[1]))
            norm_score_list.append(float(second_part[2]))

raw_score_array = np.array(raw_score_list).reshape([-1, 1])
plot(raw_score_array, 'raw_score')