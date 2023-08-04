import os
import numpy as np
import pandas as pd



def get_df_by_date(date):
    file_name = os.path.join('./data', f'{date}.log')
    lines = []
    with open(file_name, 'r') as f:
        for line in f:
            if line.startswith("doc_id"):
                sp = line.split(",")
                fields = dict()
                for i in sp:
                    ssp = i.split('=')
                    if len(ssp) == 2:
                        fields[ssp[0].strip()] = ssp[1].strip()
                if len(fields) >= 4:
                    lines.append((fields['doc_id'], fields['follow'], fields['click'], fields['w']))

    df = pd.DataFrame(data=lines, columns=['doc_id', 'follow', 'click', 'w'])
    df['w'] = df['w'].astype(np.float)
    return df


if __name__ == '__main__':
    df1 = get_df_by_date('0902')