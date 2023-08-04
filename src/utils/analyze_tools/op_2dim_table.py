from typing import List
from pandas import DataFrame


def pivot(df, index, columns, values):
    pdf = df.pivot(index='粉丝量级', columns='日期', values='内容真实曝光pv')

    def sort_by_level(sl):
        if sl == '0':
            return -1
        aa = sl.split(',')
        aa = aa[0].strip('(')
        if aa.endswith('w'):
            aa = aa.replace('w', '0000')
        elif aa.endswith('k'):
            aa = aa.replace('k', '000')
        return int(aa)

    pdf = pdf.reset_index()
    pdf['row'] = pdf['粉丝量级'].apply(lambda x: sort_by_level(x))
    pdf2 = pdf.sort_values('row')


def cal_percent_by_cols(df: DataFrame,
                        columns: List[str],
                        trans_col_func = None,
                        replace = True,
                        save = False):
    for col in columns:
        if trans_col_func:
            df[col] = df[col].apply(lambda x: trans_col_func(x))
        s = df[col].astype(float).sum()
        if replace:
            df[col] = df[col].apply(lambda x: x / s).apply(lambda x: format(x, '.2%'))
        else:
            df[col + '_percent'] = df[col].apply(lambda x: x / s).apply(lambda x: format(x, '.2%'))

    if save:
        df.to_csv('result.csv', sep='\t')
    return df
    # df[].iloc[0:10].sum()