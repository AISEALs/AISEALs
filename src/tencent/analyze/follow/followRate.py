import pandas as pd
import numpy as np


df = pd.read_csv("/Users/jiananliu/Downloads/bfc26264-5550-4ee7-85bc-ca1ebd74a789.csv")
df.columns = ['video_id', 'follow_seven_day']

df['follow_seven_day'] = pd.to_numeric(df['follow_seven_day'], errors='coerce').astype(np.float)

df2 = df.drop_duplicates(keep='last')