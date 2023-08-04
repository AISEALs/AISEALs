import pandas as pd


df = pd.read_csv('data/一二线城市大发文占比-feeds短视频.csv', sep='\t')

df = df.pivot('publish_doc_level', 'is_first_tier_city', 'expose').reset_index()

df.to_csv('result.csv', sep='\t')