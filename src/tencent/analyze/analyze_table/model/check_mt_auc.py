from collections import defaultdict
import pandas as pd


metrics_dict = defaultdict(dict)

with open('check_mt_auc.txt', 'r') as f:
    for i in f:
        date_time = ''
        sp = i.strip().split('|')
        for kv in sp:
            try:
                kv = kv.strip()
                if kv == '':
                    continue
                if 'data_time' in kv:
                    k = kv.split(':')[0].strip()
                    v = kv.split(':')[1].strip('\' ')
                    date_time = v
                else:
                    print(kv)
                    k = kv.split('=')[0].strip()
                    v = float(kv.split('=')[1].strip())
                    metrics_dict[date_time][k] = v
                    print(metrics_dict)
            except:
                print(kv)

cols = ['report_sample_num', 'commenttime_loss', 'report_sample_nuwcomment_lossm', 'feedback_auc', 'predict_avg', 'real_avg', 'copc', 'feedback_loss', 'follow_auc', 'follow_loss', 'praise_auc', 'praise_loss', 'rcomment_auc', 'rcomment_loss', 'share_auc', 'share_loss', 'wcomment_auc', 'wcomment_loss']
df = pd.DataFrame([[k] + [v[col] for col in cols] for k, v in metrics_dict.items()],
                   columns=['dateHour'])