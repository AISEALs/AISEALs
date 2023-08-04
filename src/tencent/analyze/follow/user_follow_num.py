import datetime
import pandas as pd
from src.utils.dater import date_range


total_df = pd.read_csv("/Users/jiananliu/Desktop/work/tencent/analyze/table/follow/user_follow_num/20210721-0729.txt", sep='\t')


start_date = datetime.date(2021, 7, 21)
end_date = datetime.date(2021, 7, 29)
date_list = [int(d) for d in date_range(start_date, end_date)]

for d in date_list:
    df = total_df[total_df.ds2 == d]
    total_follow_num = df.follow_num.sum()
    df = df.sort_values('follow_num', ascending=False)

    df.follow_num.describe()
    quantile = df.follow_num.quantile(0.997)
    topn_follow_num = df[df.follow_num > quantile].follow_num.sum()

    print(f'----date: {d}-----')
    print(f'total_follow_num: {total_follow_num}')
    # print(f'topn_follow_num: {topn_follow_num}')
    print(f'top 3/1000 follow num: {quantile}, max follow num: {df.follow_num.max()}')
    print('{:.2%}'.format(topn_follow_num/total_follow_num))
