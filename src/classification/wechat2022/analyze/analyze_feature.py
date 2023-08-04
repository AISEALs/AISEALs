import pandas as pd
import json
import sys
sys.path.append("../")
from category_id_map import category_id_to_lv1id, category_id_to_lv2id


def get_labels_df():
    with open('./data/annotations/labeled.json') as f:
        features = json.load(f)

    df = pd.DataFrame(features)

    df['title_length'] = df['title'].apply(lambda x: len(x))
    df['ocr_length'] = df['ocr'].apply(lambda x: len(x))
    df['lv1id'] = df['category_id'].apply(lambda x: category_id_to_lv1id(x))
    df['lv2id'] = df['category_id'].apply(lambda x: category_id_to_lv2id(x))

    return df

if __name__ == '__main__':
    df = get_labels_df()
    print(df['ocr_length'].describe())
