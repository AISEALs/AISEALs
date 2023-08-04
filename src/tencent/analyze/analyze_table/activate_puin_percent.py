import os
import pandas as pd

path = "/Users/jiananliu/Desktop/work/tencent/analyze/table/activate_puin_percent/"
# scene = 'mini'
df = pd.read_csv(os.path.join(path, "20210624.csv"), sep=',')

df.columns = ['scene', 'puin_activate_status', 'expose']
df = df[df.eval('(scene == "mini") or (scene == "short")')]

total_expose_sum = df['expose'].sum()
df['expose_per_total'] = df.eval(f'expose*1.0/{total_expose_sum}')
df['expose_per_total'] = df['expose_per_total'].apply(lambda x: format(x, '.2%'))
df['total_expose_scene'] = df.groupby('scene')['expose'].transform(sum)
df['expose_per_scene'] = df['expose']/df['total_expose_scene']
df['expose_per_scene'] = df['expose_per_scene'].apply(lambda x: format(x, '.2%'))

df.to_csv(os.path.join(path, 'save.csv'))

