import sys
import math
import numpy as np

home_path = sys.argv[1]
model_version = sys.argv[2]
pre_name = home_path + "/data/one_step_tmp/part-00000"
f_name = home_path + "/data/one_step_tmp/part-00001"
f = open(pre_name, "r")
f2 = open(f_name, "w")
reword_seg = []
for line in f:
    line = line.strip()
    value = line.split("|")
    if len(value) != 5:
        continue
    value.pop()
    # click = float(value[1].split("_")[0]) / 12.0
    labels = value[1].split("_")
    if len(labels) != 6:
        continue
    playtime = float(labels[0])
    isFinish = float(labels[1])
    playRatio = float(labels[2])
    skip = float(labels[3])
    flpush_num = float(labels[4])
    reward = float(labels[5])
    if math.isnan(reward):
        reward = 0.0    
    # 2021 12 03 10:12
    #playRatio = min(playRatio, 20) / 20
    #playtime = min(playtime, 1200) / 1200
    #playtime = max(0.0001, playtime)
    #playtime = min(math.log(playtime), 7.5) / 7.5
    #reword = pow(playRatio, 1) * pow(playtime, 1) # 2021-12-10 16:32 
    #reword = pow(playtime, 1) # 2021-12-06 11:32
    reword = reward 
    now_model_version = value[2]
    rand_num = np.random.uniform(0, 1)
    if (now_model_version == model_version):
        value[1] = str(reword) + "@" + str(playtime)
        f2.write("\t".join(value) + "\n")
f.close()
f2.close()

d = {}
f = open(f_name, "r")
for line in f:
    line = line.strip()
    value = line.split("\t")
    if len(value) < 4:
        continue
    for i in range(3, len(value) - 1):
        value2 = value[i].split(",")
        if len(value2) != 3:
            continue
        fea = value2[0]
        d[fea] = 1
f.close()

n = len(d)
print("fea_num:",n)
i = 0
d_dump = []
for item in d:
    d_dump.append(item)
    d[item] = i
    i = i + 1

# dic = {'ES_VIDEO_CTR_BETA_27_0':'', 'ES_VIDEO_TIME_BETA_27_0':'','ES_MINIVIDEO_CTR_BETA_27_0':'','ES_MINIVIDEO_TIME_BETA_27_0':'','ES_NEWS_CTR_BETA_27_0':'','ES_NEWS_TIME_BETA_27_0':'', 'ES_VIDEO_CTR_BETA_26_0':'', 'ES_VIDEO_TIME_BETA_26_0':'','ES_MINIVIDEO_CTR_BETA_26_0':'','ES_MINIVIDEO_TIME_BETA_26_0':'','ES_NEWS_CTR_BETA_26_0':'','ES_NEWS_TIME_BETA_26_0':'', 'ES_VIDEO_CTR_BETA_25_0':'', 'ES_VIDEO_TIME_BETA_25_0':'','ES_MINIVIDEO_CTR_BETA_25_0':'','ES_MINIVIDEO_TIME_BETA_25_0':'','ES_NEWS_CTR_BETA_25_0':'','ES_NEWS_TIME_BETA_25_0':'','ES_VIDEO_CTR_BETA_24_0':'', 'ES_VIDEO_TIME_BETA_24_0':'','ES_MINIVIDEO_CTR_BETA_24_0':'','ES_MINIVIDEO_TIME_BETA_24_0':'','ES_NEWS_CTR_BETA_24_0':'','ES_NEWS_TIME_BETA_24_0':'','ES_VIDEO_TIME_BETA_20_300':'','ES_VIDEO_CTR_BETA_20_300':'','ES_MINIVIDEO_TIME_BETA_20_300':'','ES_MINIVIDEO_CTR_BETA_20_300':'','ES_NEWS_TIME_BETA_20_300':'','ES_NEWS_CTR_BETA_20_300':'','ES_VIDEO_TIME_BETA_21_300':'','ES_VIDEO_CTR_BETA_21_300':'','ES_MINIVIDEO_TIME_BETA_21_300':'','ES_MINIVIDEO_CTR_BETA_21_300':'','ES_NEWS_TIME_BETA_21_300':'','ES_NEWS_CTR_BETA_21_300':'','ES_VIDEO_TIME_BETA_22_300':'','ES_VIDEO_CTR_BETA_22_300':'','ES_MINIVIDEO_TIME_BETA_22_300':'','ES_MINIVIDEO_CTR_BETA_22_300':'','ES_NEWS_TIME_BETA_22_300':'','ES_NEWS_CTR_BETA_22_300':'','ES_VIDEO_TIME_BETA_23_300':'','ES_VIDEO_CTR_BETA_23_300':'','ES_MINIVIDEO_TIME_BETA_23_300':'','ES_MINIVIDEO_CTR_BETA_23_300':'','ES_NEWS_TIME_BETA_23_300':'','ES_NEWS_CTR_BETA_23_300':''}
# dic2 = {}
f2 = open(home_path + "/data/one_step_tmp/allsample.txt", "w")
f = open(f_name)
'''
f3 = open(home_path + "/data/one_step_tmp/ESTIME12.txt", "w")
filter_file = open(home_path + "/data/all_theta_dict.txt", 'r')
filter_dic = {}
for line in filter_file:
	line = line.strip().split('\t')
	if len(line) != 2:
		continue
	filter_dic[line[0]] = float(line[1])
'''
SLOT = set(["1", "2", "3", "4", "5", "8", "9", "10", "11", "12", "13", "16", "17", "18", "19", "20" ,"21", "22", "23","24"])
count = 0
for line in f:
    filter_flag = 0
    line = line.strip()
    value = line.split("\t")
    if len(value) < 4:
        continue
    key = value[0]
    reword = value[1]
    flag = 0
    theta = [0.0 for i in range(0, n)]
    noise = [0.0 for i in range(0, n)]
    for i in range(3, len(value)):
        value2 = value[i].split(",")
        if len(value2) != 3:
            continue
        if value2[0] not in d:
            continue
        fea_sign = value2[0]
        fea_slot_key = fea_sign.split("_")
        if len(fea_slot_key) < 5:
            continue
        slot = fea_slot_key[4]
        if slot not in SLOT:
            #print("NOT in Slot:", value2)
            continue
        # if (filter_dic[value2[0]] - float(value2[1])) < 0.00001:
        #	filter_flag += 1
        k = d[value2[0]]

        theta[k] = float(value2[1])
        noise[k] = float(value2[2])

    theta_str = [str(item) for item in theta]
    noise_str = [str(item) for item in noise]
    ret = "\t".join([key, ",".join(theta_str), ",".join(noise_str), reword])
    '''
    if filter_flag > 200:
        f2.write(ret + "\n")
    else:
        count += 1
    '''
    f2.write(ret + "\n")
    count += 1
print("sample_num",count)
f.close()
f2.close()

# dump
f = open(home_path + "/data/one_step_tmp/fea_list.txt", "w")
f.write("\t".join(d_dump) + "\n")
f.close()

f = open(home_path + "/data/sample_fea_list_record.txt", "a")
f.write("\t".join(d_dump) + "\n")
f.close()
