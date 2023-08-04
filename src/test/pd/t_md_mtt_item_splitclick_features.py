import os
import numpy as np
import pandas as pd


pd.options.mode.chained_assignment = None


base_dir = '/Users/jiananliu/Desktop/work/tencent/analyze/table/t_md_mtt_item_splitclick_features'

if __name__ == '__main__':
    file_path = os.path.join(base_dir, '20201215.csv')
    df = pd.read_csv(file_path, sep='\t')

    df.columns = df.columns.map(lambda x: x.replace('t_md_mtt_item_splitclick_features.', ''))

    scene_df = df[df.eval('scene == 3')]

    scene_df.loc[:, 'cl_max'] = scene_df.groupby(['item_id'], sort=False)['view_level'].transform(max)

    g100_df = scene_df[scene_df.eval('view_level <= 110 and cl_max >= 100')]

    pdf = pd.pivot(g100_df, index='item_id', columns='view_level')

    # pdf[pdf.eval('item_id == 8992555866526442926')]