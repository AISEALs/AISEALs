import pandas as pd
import matplotlib.pyplot as plt


file_name = '/Users/jiananliu/Desktop/work/AISEALs/src/tencent/analyze/highfollowadjust/data/1210-1220小时级转粉率+高转粉内容占比.xlsx'
df = pd.read_excel(file_name)
print (df)
df['ds'] = df['ds'].astype(str)

df['dt'] = pd.to_datetime(df.ds, format='%Y%m%d%H')

df.columns = ['ds', 'cnt_vv', 'follow_num', 'follow_ctr', 'follow_ctr_day', 'dt']

time_list=df['ds'].tolist()
step = 24

df.plot(x='ds', y='follow_ctr')
plt.xticks(range(0,len(time_list), step), [time_list[i] for i in range(0,len(time_list),1) if i%step==0], rotation=-90)
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.3)
plt.show()