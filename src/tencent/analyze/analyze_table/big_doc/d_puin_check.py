import pandas as pd


file_name = "./data/D级账号check.csv"

df = pd.read_csv(file_name, sep=',')
df = df[df.scene == 4]
print(df)

