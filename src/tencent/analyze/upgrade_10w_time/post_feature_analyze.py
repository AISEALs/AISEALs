import os
import pandas as pd
import numpy as np
from texttable import Texttable

base_dir = '/Users/jiananliu/Desktop/work/tencent/analyze/'

if __name__ == '__main__':
    file_name = os.path.join(base_dir, 'result/short_3day_result.csv')
    df = pd.read_csv(file_name)
    print(df)
    # df.fillna(0, inplace=True)
    df.rename(columns={'视频质量评估': 'result'}, inplace=True)
    df['result'].replace('A', 'a', inplace=True)
    df['result'].replace('B', 'b', inplace=True)

    df_label = df[~df.result.isna()]

    print(df_label.describe())

    print(df_label.groupby('result').size())
    # print(df_label.columns)

    # 'attention_percent',
    feature_columns = ['floatClick', 'finish_read_90_percent',
                       'skip_read_10_percent', 'praise_percent', 'ctr',
                       'share_percent', 'comment_percent', 'see_comment_percent']

    for label in feature_columns:
        data = []
        # label = 'praise_percent'
        print(f'{label}:')
        for (start, end) in [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]:
            row = {'quantile': f'{start}~{end}'}
            start_split = df_label[label].quantile(start)
            end_split = df_label[label].quantile(end)
            tmp_df = df_label[df_label.eval(f'{label} >= {start_split} and {label} < {end_split}')]
            result_df = tmp_df.groupby('result').count()[label] / len(tmp_df)
            row.update(result_df.to_dict())
            data.append(row)

            # result_df = pd.DataFrame(data)
            # tb.header(result_df.columns)

        result_df = pd.DataFrame(data)
        tb = Texttable()
        tb.set_precision(2)
        tb.set_max_width(0)
        # tb.set_cols_align(['l'] + ['l'] * 9)
        # tb.set_cols_dtype(['t'] + ['t'] * 9)
        tb.header(['quantile', 's', 'a', 'b', 'c'])
        tb.add_rows(result_df.values, header=False)
        print(tb.draw())
