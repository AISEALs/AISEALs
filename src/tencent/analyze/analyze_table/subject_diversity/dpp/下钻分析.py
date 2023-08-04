import xlrd
import sys

# tab_data_path = sys.argv[1]
# exp_name = sys.argv[2]

tab_data_path = r'/Users/guohao/Downloads/视频浮层冷启加权实验v1（exp_qb_float_mini_video_cold_start_weight_v1）（实验名称）- 累计（下钻）结果数据.csv'
exp_name = "视频浮层冷启加权实验v1"

wb = xlrd.open_workbook(tab_data_path)
sheet_names = wb.sheet_names()
sheet_name = ""
for value in sheet_names:
    if exp_name in value:
        sheet_name = value
        break

if not sheet_name:
    raise ValueError("no sheet_name, check exp_name!")
sheet = wb.sheet_by_name(sheet_name)
print('sheet的名称：', sheet.name)

need_data_name = ['人均信息流商业化收入（元）', '总时长', '真实总点击次数', '优质DAU', '消费DAU', 'DAU', '视频总时长', '真实视频点击PV', '后台视频渗透率']
active_freq_dimension = ['低频', '中频', '高频']

output_data = {}

for rown in range(3, sheet.nrows):
    row_data = sheet.row_values(rown)
    data_name = row_data[0]
    data_freq_dim = row_data[1]
    data_relative_diff = row_data[7]
    data_p_value = row_data[9]
    data_p_significant = row_data[10]

    data_name = '真实总点击次数' if data_name == '总消费pvvv' else data_name
    data_name = '总时长' if data_name == '信息流总时长' else data_name
    data_name = '优质DAU' if data_name == '优质DAU累计去重用户数' else data_name
    data_name = '消费DAU' if data_name == '消费DAU累计去重用户数' else data_name
    data_name = 'DAU' if data_name == '信息流DAU累计去重用户数' else data_name
    data_name = '视频总时长' if data_name == '视频播放时长' else data_name
    data_name = '真实视频点击PV' if data_name == '视频VV' else data_name
    data_name = '后台视频渗透率' if data_name == '视频渗透率' else data_name
    data_name = '人均信息流商业化收入（元）' if data_name == '人均信息流收入（AMS+DSP）' else data_name

    output_key = (data_freq_dim + '_' + data_name).upper()
    output_value = data_name + ':' + data_relative_diff + '(' + data_p_significant + ')'
    output_data[output_key] = output_value

the_lack_name = []
for dim in active_freq_dimension:
    print('\n' + dim + ':')
    for name in need_data_name:
        output_key = dim + '_' + name
        if output_key in output_data:
            print(output_data[output_key])
        else:
            the_lack_name.append(output_key)
print('lack_names: ', the_lack_name)