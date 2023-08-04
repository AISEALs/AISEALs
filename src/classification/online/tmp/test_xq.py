file_name = "/Users/jiananliu/Downloads/心情_情绪表达_部落21-25日数据.txt"

import pandas as pd
doc_df = pd.read_csv(file_name, sep='\t', header=None, encoding='utf-8')
doc_df.columns = ["id", "line"]
doc_df = doc_df.dropna()

from fastText import load_model
model = load_model("/Users/jiananliu/work/AISEALs/src/text_classification/tribe_labels/model/fasttext/fasttext_model_心情_情绪_想法表达.bin")

results = model.precision(doc_df.line.tolist())
doc_df['predict'] = results[0]
doc_df['predict'] = doc_df['predict'].map(lambda x: x[0])
doc_df['prob'] = results[1]

