
tests = '''
    all:1_2_2_2_10000_10000_10000_10_10000_10000_3_0_0_4_0_7_0_-1_-0.05_0_0.75_1_20_38_-18
    # 4022987:1_2_2_2_10000_10000_10000_10_10000_10000_3_0_0_5_0_7_0_-1_-0.05_0_0.75_1_20_38_-18
    # 4106407:1_2_2_2_10000_10000_10000_10_10000_10000_3_0_0_5_0_7_0_-1_-0.05_0_0.5_1_20_38_-18
    # 4106406:1_2_2_2_10000_10000_10000_10_10000_10000_3_0_0_5_0_7_0_-1_-0.04_0_0.6_1_40_78_-38
    # 4106405:1_2_2_2_10000_10000_10000_10_10000_10000_3_0_0_5_0_7_0_-1_-0.04_0_0.8_1_40_38_-13
    # 4106404:1_2_2_2_10000_10000_10000_10_10000_10000_1_0_0_5_0_7_0_-1_-0.04_0_1.0_1_40_78_-38
    # 4106403:1_2_2_2_10000_10000_10000_10_10000_10000_1_0_0_5_0_7_0_-1_-0.04_0_1.0_1_20_38_-18
    '''

aa_list = list(tests.strip().split("\n"))


for aa in aa_list:
    print('-' * 20)
    sp = aa.strip().split(":")
    print(aa.strip())

    vParams = sp[1].split("_")
    print(len(vParams))
    print(f'isDebug:{vParams[0]}')
    print(f'puin1DayFollowBias:{vParams[1]}')
    print(f'puin7DayFollowBias:{vParams[2]}')
    print(f'puin30DayFollowBias:{vParams[3]}')
    print(f'puin1DayExposureBias:{vParams[4]}')
    # param.puin7DayExposureBias = TC_Common::strto < double > (vParams[5]);
    # param.puin30DayExposureBias = TC_Common::strto < double > (vParams[6]);
    # param.puin_follow_min_threshold_ = TC_Common::strto < int64_t > (vParams[7]);
    # param.puin_exposure_min_threshold_ = TC_Common::strto < int64_t > (vParams[8]);
    # param.item_exposure_min_threshold_ = TC_Common::strto < int64_t > (vParams[9]);
    print(f'item_exposure_min_threshold_: {vParams[9]}')
    print(f'follow_min_threshold_: {vParams[10]}')
    print(f'tag_num_threshold_: {vParams[11]}')
    # param.first_use_days_threshold_ = TC_Common::strto < int64_t > (vParams[12]);
    print(f'param.mode_: {vParams[13]}')
    print(f'use_crowd_follow_ctr_: {vParams[14]}')
    # param.top_n_ = -1;
    # param.follow_num_weight_ = 1;
    # param.follow_rank_weight_ = 1;
    # param.follow_time_weight_ = 1;
    s = 15
    print(f's_sub_param.mode_ = {vParams[s + 0]}')
    print(f's_sub_param.follow_ctr_min_threshold_ = {vParams[s + 1]}')
    print(f's_sub_param.follow_ctr_max_threshold_ = {vParams[s + 2]}')
    print(f's_sub_param.item_follow_weight_ = {vParams[s + 3]}')
    print(f's_sub_param.puin_follow_weight_ = {vParams[s + 4]}')
    print(f's_sub_param.a_weight_ = {vParams[s + 5]}')
    print(f's_sub_param.min_weight_ = {vParams[s + 6]}')
    print(f's_sub_param.max_weight_ = {vParams[s + 7]}')
    print(f's_sub_param.alpha_ = {vParams[s + 8]}')
    print(f's_sub_param.beta_ = {vParams[s + 9]}')

    if int(vParams[13]) in [4, 5]:
        print(f'{sp[0]}: 1/(1+e^({vParams[s+3]}*x+{vParams[s+5]}))*{vParams[s+8]}{vParams[s+9]} in {int(vParams[s+6]), int(vParams[s+7])}, follow_min_thre:{vParams[10]}')
    elif int(vParams[13]) == 3 and int(vParams[s+0]) == 2:
        print(f'{sp[0]}:{vParams[s+5]}*x in {int(vParams[s+6]), int(vParams[s+7])}')


    #  1/(1 + e^ (x *item_follow_weight_ + a_weight_)) * alpha + beta
    #  1/(1 + e^ (x * -0.05 + 0.5 ) * 14 - 6
