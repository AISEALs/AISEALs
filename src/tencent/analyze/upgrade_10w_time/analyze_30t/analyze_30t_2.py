import os
import re
import pandas as pd


base_dir = '/Users/jiananliu/Desktop/work/tencent/analyze/short_time/analyze_30t'


scene = 3
version = '20201202'
if scene == 3 or scene == 4:
    file_name = f'video_{version}.csv'
else:
    file_name = f'feeds_{version}.csv'

item_id = 'doc_id' if scene == 1 else 'video_id'


def save_file(df, file_name):
    file_path = f'{base_dir}/{file_name}'
    df.to_csv(file_path, index=False)
    print(f'save to {file_path}')


def get_features_df():
    feature_file_name = os.path.join(base_dir, "garlandliu_sptz_1606190342.csv")
    feature_df = pd.read_csv(feature_file_name, sep=',')

    feature_df['qbdocid'] = feature_df['qbdocid'].apply(lambda x: int(re.sub('["=]', '', x)))
    feature_df = feature_df.drop(['原始docid'], axis=1)

    return feature_df

def get_vv_df():
    feature_file_name = os.path.join(base_dir, "10w-vv-cal-2671061-20201125224641.csv")
    vv_df = pd.read_csv(feature_file_name, sep='\t')

    if scene != 1:
        vv_df = vv_df[vv_df.eval(f'scene == {scene}')]

    vv_df.dropna(inplace=True)

    return vv_df


def join(df, vv_df, feature_df = None):
    df.drop(['vv', 'scene'], axis=1, inplace=True)
    join_df = df.join(vv_df.set_index(item_id), on='id', how='left')

    if 'url' in join_df.columns:
        join_df = join_df[~join_df['url'].str.contains('Android', na=False)]

    if feature_df is not None:
        join_df = join_df.join(feature_df.set_index('qbdocid'), on='id', how='left')

    final_df = join_df.reset_index(0)
    return final_df

def print_distribution_not_release(df):
    for idx, join_df in df.groupby('cur_date'):
        print('-' * 20)
        for ptype in ['mean', 'p95']:
            if ptype == 'p95':
                q50_take_hour = join_df['take_hour'].quantile(0.95)
            else:
                q50_take_hour = join_df['take_hour'].mean()
            print(str(idx) + '  ' + ptype + ':')
            for i in [0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]:
                q_v = join_df['take_hour'].quantile(i)

                filter_join_df = join_df[join_df.eval(f'take_hour > {q_v}')]

                not_filter_join_df = join_df[join_df.eval(f'take_hour <= {q_v}')]

                if ptype == 'p95':
                    q_not_filter = not_filter_join_df["take_hour"].quantile(0.95)
                else:
                    q_not_filter = not_filter_join_df["take_hour"].mean()

                filter_sum = filter_join_df["T1_expose"].sum()
                total_sum = join_df["T1_expose"].sum()

                # print(filter_sum, total_sum)
                print(
                    f'卡耗时{int(i * 100)}分位(耗时: {"%.2f" % q_v}) 前后{ptype}耗时变化：{"%.2f" % q50_take_hour}->{"%.2f" % q_not_filter}h 减少耗时占比{"%.2f" % ((q50_take_hour - q_not_filter) / q50_take_hour)}, 减少消费占比: {"%.4f" % (filter_sum / total_sum)}')

def read_table():
    df = pd.read_csv(os.path.join(base_dir, 'table.txt'), sep='|')
    return df[['video_id', 'total_count', '95w']]

def read_doc(idx_list, ftype=0, short_v=0, filter=False):
    file_suffix = 'video' if ftype == 0 else 'jingpin'
    if isinstance(idx_list, list):
        df_list = []
        for idx in idx_list:
            file_name = f'{file_suffix}_{idx}.csv'
            file_path = os.path.join(base_dir, file_name)
            df = pd.read_csv(file_path, sep=',')
            print(idx, df)
            df.loc[:, 'thre_date'] = df.eval('vv_ftime/100').astype(int)
            df = df[df.eval(f'thre_date == {idx}')]
            df_list.append(df)
        df = pd.concat(df_list, axis=0, ignore_index=True)
        idx = f'{idx_list[0]}~{idx_list[-1]}'
    else:
        idx = idx_list
        file_name = f'{file_suffix}_{idx}.csv'
        file_path = os.path.join(base_dir, file_name)
        df = pd.read_csv(file_path, sep=',')
        df.loc[:, 'thre_date'] = df.eval('vv_ftime/100').astype(int)
        # print(idx)
        # print(df.groupby('thre_date').count()['id'])
        if ftype == 0:
            df = df[df.eval(f'thre_date == {idx}')]
    df = df[df.eval(f"short_v == {short_v}")]

    # df['take_hour'] = df['cost_from_input']*1.0/3600
    df.loc[:, 'take_hour'] = df.loc[:, 'cost_from_input']*1.0/3600

    # todo: 为什有\N
    df.loc[df['vv_daily'] == '\\N', 'vv_daily'] = 0
    df.loc[df['vv_daily'].isna(), 'vv_daily'] = 0
    df.loc[:, 'vv_daily'] = df['vv_daily'].astype('int')

    print(ftype, ":")
    if ftype == 0:
        if short_v == 0:
            threholds = 270
            print('短视频：')
        elif short_v == 1:
            threholds = 380
            print('小视频：')
        else:
            threholds = 50
            print('图文：')
    else:
        if short_v == 0:
            threholds = 270
            print('短视频：')
        elif short_v == 1:
            threholds = 380
            print('小视频：')
        else:
            threholds = 15
            print('图文：')

    import datetime
    # datetime.datetime.strptime(oneDay_str, "%Y-%m-%d %H:%M:%S")
    df.loc[:, 'input_ts_s'] = df['input_ts_t'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())
    # start_ts = 1606449600 - 261.9*3600
    # join_df = df[df.eval(f'input_ts_s > {start_ts}')]
    print('-' * 20)
    print(idx)
    # for ptype in ['mean', 'p95']:
    today_df = df
    # old_df = today_df[today_df.eval(f'input_ts_s < {start_ts}')]
    # print(','.join(map(str, old_df['scid'].values)))

    today_bigger_df = today_df[today_df.eval(f'take_hour >= {threholds}')]
    print('爆款一共:', today_df['id'].count(), f'大于{threholds}h的id：',  today_bigger_df['id'].count())
    print(','.join(map(str, today_bigger_df['scid'].values)))

    # view_df = today_bigger_df[['scid', 'take_hour', 'input_ts_t', 'vv_daily', 'vv', 'vv_ftime']]
    # print(','.join(map(str, today_bigger_df['scid'].values)))
    print(f'{idx} 全部10w新增数量：{"%d" % today_df["id"].count()}, p95耗时：{"%.2f" % today_df["take_hour"].quantile(0.95)}h')
    print(f'{idx} 全部10w新增数量：{"%d" % today_df["id"].count()}, mean耗时：{"%.2f" % today_df["take_hour"].mean()}h')
    today_little_df = today_df[today_df.eval(f'take_hour < {threholds}')]
    print(f'{idx} 过滤超过{threholds}h的10w新增数量：{"%d" % today_little_df["id"].count()}, p95耗时：{"%.2f" % today_little_df["take_hour"].quantile(0.95)}h')
    print(f'{idx} 过滤超过{threholds}h的10w新增数量：{"%d" % today_little_df["id"].count()}, mean耗时：{"%.2f" % today_little_df["take_hour"].mean()}h')
    # if filter:
    #     return today_bigger_df
    # else:
    #     # print(today_df['take_hour'].describe())
    #     return today_df
    # return today_bigger_df['scid'].values
    return df['scid'].values

def tst():
    table_df = read_table()
    table_df.columns = ['video_id', 'expose_num', 'ts_95w']
    df_01 = read_doc("20201130", filter=True)
    df_02 = read_doc("20201201", filter=True)
    df = pd.concat([df_01, df_02], ignore_index=True)
    df = df.join(table_df.set_index('video_id'), on='scid')
    df['take_hour_95'] = df.eval('ts_95w - input_ts_s') * 1.0 / 3600
    bb = df[df.eval('take_hour_95 > 261.9')]

    aa = pd.read_csv(os.path.join(base_dir, 'test-2697938-20201203210900.csv'), sep='\t')
    print(aa[aa.eval("video_id == 2867656382886847851")])


if __name__ == '__main__':
    # print_distribution_not_release(df)

    ids_list = []
    ftype = 0 # 0:优质， 1：竞品
    short_v = 2 # 0: short, 1: min, 2: 图文
    ids_list.extend(read_doc("20201201", ftype, short_v))
    ids_list.extend(read_doc("20201202", ftype, short_v))
    ids_list.extend(read_doc("20201203", ftype, short_v))
    ids_list.extend(read_doc("20201204", ftype, short_v))
    ids_list.extend(read_doc("20201205", ftype, short_v))
    ids_list.extend(read_doc("20201206", ftype, short_v))
    ids_list.extend(read_doc("20201207", ftype, short_v))
    ids_list.extend(read_doc("20201208", ftype, short_v))
    ids_list.extend(read_doc("20201209", ftype, short_v))
    ids_list.extend(read_doc("20201210", ftype, short_v))
    ids_list.extend(read_doc("20201211", ftype, short_v))
    ids_list.extend(read_doc("20201212", ftype, short_v))
    ids_list.extend(read_doc("20201213", ftype, short_v))
    # read_doc("20201209", ftype, short_v)
    # read_doc(['20201201', '20201202', '20201203', '20201204', '20201205', '20201206', '20201207', '20201208'], ftype, short_v)
    # read_doc(['20201201', '20201202', '20201203', '20201204'], ftype, short_v)
    # read_doc(['20201205', '20201206', '20201207', '20201208', '20201209', '20201210', '20201211', '20201212', '20201213'], ftype, short_v)

    # print(set(ids_list))

