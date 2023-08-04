import os
import pandas as pd

base_dir = '/Users/jiananliu/Desktop/work/tencent/analyze/'
date_dir = 'short_time'
# data_type = '297748_20201103_mini'
# data_type = '297748_20201103_mini_real_click'
gray_id = 303661
data_type = 'feeds_10w-2683411-20201130143343'
# gray_id = 307660
# data_type = '效果统计-2669618-20201124200702'
# gray_id = 288794
# data_type = 'account_adjust_exposure_mini'

# 小视频实验2 6号上线，增加6-10w
# gray_id = 307656
# data_type = '307656_20201107_mini'


debug_mode = False


def read_evaluate_file():
    file_name = os.path.join(base_dir, date_dir, f"{data_type}.csv")
    df = pd.read_csv(file_name, sep='\t')

    # df.columns = ['ds', 'a', 'b']

    return df


if __name__ == '__main__':
    df = read_evaluate_file()

    scene = 4

    # df.eval('increase=(show_pv_297748-show_pv_297747)/show_pv_297747*100.0', inplace=True)

    # -- 短视频 6号到7号发布的，后台曝光
    experimental_group = f'show_pv_{gray_id}'
    control_group = f'show_pv_{gray_id-1}'
    df.eval(f'increase=({experimental_group}-{control_group})/{control_group}*100.0', inplace=True)
    print(df)
    # test1_total_pv = df[experimental_group].sum()
    # test2_total_pv = df[control_group].sum()
    # increase = (test1_total_pv - test2_total_pv)*1.0/test2_total_pv * 100
    # print(f"total实验组总曝光pv：{test1_total_pv}, 对照组总曝光pv：{test2_total_pv}, 增加占比：{'%.2f' % increase}%")
    #
    # sab_num1 = df[~df.eval('c3 == "2_level" or c3 == "1_level"')][experimental_group].sum()
    # sab_num2 = df[~df.eval('c3 == "2_level" or c3 == "1_level"')][control_group].sum()
    # increase3 = (sab_num1 - sab_num2)*1.0/sab_num2*100
    # print(f"sab实验组总曝光pv：{sab_num1}, 对照组总曝光pv：{sab_num2}, 增加占比：{'%.2f' % increase3}%")
    #
    # not_sab_num_num1 = df[df.eval('c3 == "2_level" or c3 == "1_level"')][experimental_group].sum()
    # not_sab_num_num2 = df[df.eval('c3 == "2_level" or c3 == "1_level"')][control_group].sum()
    # increase2 = (not_sab_num_num1 - not_sab_num_num2)*1.0/not_sab_num_num2 * 100
    # print(f"c_d实验组总曝光pv：{not_sab_num_num1}, 对照组总曝光pv：{not_sab_num_num2}, 增加占比：{'%.2f' % increase2}%")

    # print(df[['c3', 'increase']])
    # total_increase = (df['a'].sum() - df['b'].sum()) / df['b'].sum()
    # print(f'32h以内，曝光增长：{"%.2f" % (total_increase * 100.0)}%')