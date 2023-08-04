import os
import pandas as pd


base_dir = '/Users/jiananliu/Desktop/work/tencent/analyze/short_time/analyze_30t'

scene = 3
if scene == 3 or scene == 4:
    file_name = 'MTT.VideoRelatedRankServer_BaokuanNotRelease_20201130.log'
else:
    file_name = 'MTT.VideoRelatedRankServer_BaokuanNotRelease_20201130.log'

item_id = 'doc_id' if scene == 1 else 'video_id'

def filter_video_id():
    time_over_ids = []
    with open(file_path) as f:
        for line in f:
            sp = line.strip().split("|")
            vid_str = sp[2]
            vid = int(vid_str.split(":")[1])
            take_hour_str = sp[7]
            take_hour = float(take_hour_str.split(":")[1])
            thresholds_str = sp[-1]
            thresholds = float(thresholds_str.split(":")[1])
            if take_hour > thresholds and thresholds == gray_thresholds:
                print(line)
                time_over_ids.append(vid)
                if len(time_over_ids) >= 10:
                    break
    print(','.join(map(str, time_over_ids)))


if __name__ == '__main__':
    file_path = os.path.join(base_dir, file_name)

    gray_id = '337361'
    gray_thresholds = 271.6

    filter_video_id()
    # with open(file_path) as f:
    #     for line in f:
    #         sp = line.strip().split("|")
    #         vid_str = sp[2]
    #         vid = int(vid_str.split(":")[1])
    #         take_hour_str = sp[7]
    #         take_hour = float(take_hour_str.split(":")[1])
    #         thresholds_str = sp[-1]
    #         thresholds = float(thresholds_str.split(":")[1])
    #         if vid == 3642275542440263019:
    #             print(line)
