from os import listdir
from os.path import isfile, join
import json
import sys
from collections import defaultdict

# import pandas as pd


dump_folder_path = "./{}/".format(sys.argv[1])
result_dir = "./{}/".format(sys.argv[2])

for query_folder in listdir(dump_folder_path):
    print(result_dir + query_folder)
    f = open(result_dir + query_folder, "w")
    # f = sys.stdout
    f1 = open(result_dir + "result.txt", "a")
    taf_rsp_file = join(dump_folder_path, query_folder, "taf_rsp")
    trs_rsp_file = join(dump_folder_path, query_folder, "trs_rsp")
    if not isfile(taf_rsp_file) or not isfile(trs_rsp_file):
        continue

    try:
        with open(taf_rsp_file, "r", encoding='UTF-8') as taf_file:
            taf_rsp_item = json.load(taf_file)
        with open(trs_rsp_file, "r", encoding='UTF-8') as trs_file:
            trs_rsp_item = json.load(trs_file)
    except:
        continue

    f.write("total taf:" + str(len(taf_rsp_item["vData"])) + ",trs:" + str(len(trs_rsp_item["vData"])) + "\n")

    taf_recall_map = {}
    taf_pred_map = {}
    taf_filter_map_in_pred = {}
    taf_rsp_map = {}
    taf_pred_list = []
    trs_pred_list = []
    trs_recall_map = {}
    trs_pred_map = {}
    trs_filter_map_in_pred = {}
    trs_rsp_map = {}
    trs_parity_pred_map = {}
    trs_recall_merged_map = defaultdict(dict)
    taf_recall_merged_map = defaultdict(dict)
    trs_recall_merged_id_set = set()
    taf_recall_merged_id_set = set()
    trs_merged_map = defaultdict(dict)
    taf_merged_map = defaultdict(dict)
    trs_final_map = {}
    taf_final_map = {}
    trs_query_map = {}
    taf_query_map = {}
    query_keys = ['rank_recall_raw_size', 'rank_recall_video_time', 'rsp_video_time']

    for item in trs_rsp_item["vData"]:
        if "returnAllRecall" in item["mpExtInfo"]:
            trs_recall_map[item["lId"]] = item
        if "returnAllMerge" in item["mpExtInfo"]:
            trs_merged_map[item["lId"]] = item  # i.e. scoretype : id : score
        if "returnAllRecallMerge" in item["mpExtInfo"]:
            trs_recall_merged_map[item["lId"]] = item
        if "returnFinalRsp" in item["mpExtInfo"]:
            trs_final_map[item["lId"]] = item
        if 'sort_rank' in item['mpDebugInfo'] \
                and item['mpDebugInfo']['sort_rank'] == '1' \
                and 'rank_recall_raw_size' in item['mpDebugInfo']:
            for k, v in item['mpDebugInfo'].items():
                for q in query_keys:
                    if k.startswith(q):
                        trs_query_map[k] = v

    for item in taf_rsp_item["vData"]:
        if "returnAllRecall" in item["mpExtInfo"]:
            taf_recall_map[item["lId"]] = item
        if "returnAllMerge" in item["mpExtInfo"]:
            taf_merged_map[item["lId"]] = item  # i.e. scoretype : id : score
        if "returnAllRecallMerge" in item["mpExtInfo"]:
            taf_recall_merged_map[item["lId"]] = item
        if "returnFinalRsp" in item["mpExtInfo"]:
            taf_final_map[item["lId"]] = item
        if 'sort_rank' in item['mpDebugInfo'] \
                and item['mpDebugInfo']['sort_rank'] == '1' \
                and 'rank_recall_raw_size' in item['mpDebugInfo']:
            for k, v in item['mpDebugInfo'].items():
                for q in query_keys:
                    if k.startswith(q):
                        taf_query_map[k] = v

    # f.write(">>>>>>>>>>>>>>>>>>>>>>>merge item(after low ratio filter)<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
    rank_select_info = defaultdict(list)
    both_recall_id_set = trs_recall_map.keys() & taf_recall_map.keys()
    both_merged_id_set = trs_merged_map.keys() & taf_merged_map.keys()
    both_rsp_id_set = trs_final_map.keys() & taf_final_map.keys()
    f.write("recall| trs size:{}, taf size:{}, both size:{}\n".format(len(trs_recall_map), len(taf_recall_map), len(both_recall_id_set)))
    f.write("recall-merge| trs size:{}, taf size:{}, both size:{}\n".format(len(trs_merged_map), len(taf_merged_map), len(both_merged_id_set)))
    f.write("final-rsp| trs size:{}, taf size:{}, both size:{}\n".format(len(trs_final_map), len(taf_final_map), len(both_rsp_id_set)))

    ## recall
    trs_target = 'recall_filter_reason'
    taf_target = 'filterReason'
    f.write("===============recall target: {}\n".format(trs_target))
    sid_filter_reason = defaultdict(list)
    reason_cnt_map = defaultdict(lambda: [0, 0])
    for sid, item in trs_recall_map.items():
        reason = int(item["mpDebugInfo"][trs_target])
        sid_filter_reason[sid].append(reason)
        reason_cnt_map[reason][0] += 1
    for sid, item in taf_recall_map.items():
        reason = int(item['mpExtInfo'][taf_target] if taf_target in item['mpExtInfo'] else '0')
        sid_filter_reason[sid].append(reason)
        reason_cnt_map[reason][1] += 1

    for sid, reasons in sid_filter_reason.items():
        if len(reasons) != 2:
            continue
        if reasons[0] == reasons[1] or reasons[0] and reasons[1]:
            continue
        else:
            print(sid, reasons)

    trs_cnt = [0, 0]
    taf_cnt = [0, 0]
    for k, v in trs_recall_merged_map.items():
        if "mpDebugInfo" in v and "low_ratio" in v["mpDebugInfo"]:
            if v["mpDebugInfo"]["low_ratio"].split("|")[0] == "1":
                trs_cnt[1] += 1
            elif v["mpDebugInfo"]["low_ratio"].split("|")[0] == "0":
                trs_cnt[0] += 1
    for k, v in taf_recall_merged_map.items():
        if "mpDebugInfo" in v and "low_ratio" in v["mpDebugInfo"]:
            if v["mpDebugInfo"]["low_ratio"].split("|")[0] == "1":
                taf_cnt[1] += 1
            elif v["mpDebugInfo"]["low_ratio"].split("|")[0] == "0":
                taf_cnt[0] += 1
    f.write("trs low ratio:{}, taf low ratio:{}\n\n".format(trs_cnt, taf_cnt))

    f.write("===============recall query info [trs, taf]:\n")
    recall_rsp_info = defaultdict(lambda: [-1, -1])
    recall_rsp_videotime = defaultdict(lambda: [-1, -1])
    other_rsp_info = defaultdict(lambda: [-1, -1])
    for k, v in trs_query_map.items():
        if k.startswith('rank_recall_raw_size'):
            recall_rsp_info[k][0] += int(v) + 1
        elif k.startswith('rank_recall_video_time'):
            recall_rsp_videotime[k][0] += int(v) + 1
        else:
            other_rsp_info[k][0] += int(v) + 1

    for k, v in taf_query_map.items():
        if k.startswith('rank_recall_raw_size'):
            recall_rsp_info[k][1] += int(v) + 1
        elif k.startswith('rank_recall_video_time'):
            recall_rsp_videotime[k][1] += int(v) + 1
        else:
            other_rsp_info[k][1] += int(v) + 1
    f1.write(query_folder + ':\n')
    for m in [recall_rsp_info, recall_rsp_videotime, other_rsp_info]:
        for k, v in m.items():
            if v[0] != v[1] and k.startswith('rank_recall_raw_size'):
                f1.write(f"{k}\t {v}\n")

    f.write('\nfilterId, trs, taf:\n')
    for k, v in reason_cnt_map.items():
        f.write(f"{k}\t {v}\n")
    # df = pd.DataFrame.from_dict(reason_cnt_map, orient='index', columns=['trs', 'taf']).sort_index()
    # f.write(df.to_string())

    f.write("\n")

    targets = ["ratio_score", "ratio_re_score", "time_score", "time_re_score", "rcomment_score", "follow_score",
               "ensemble_score", "cascade_re_score", "interact_re_score", "skip_score", "share_score", "adjust_score",
               "baokuan_score", "merged_score", "post_detail_info"]
    other_targets = ['post_selected_reason', 'sort_rank']
    for target in targets + other_targets:
        if target not in other_targets:
            f.write("===============merge target: {}\n".format(target))
        for sid in both_merged_id_set:
            taf_score = taf_merged_map[sid]["mpDebugInfo"][target] if "mpDebugInfo" in taf_merged_map[sid] and target in \
                                                                      taf_merged_map[sid]["mpDebugInfo"] else -1
            trs_score = trs_merged_map[sid]["mpDebugInfo"][target] if "mpDebugInfo" in trs_merged_map[sid] and target in \
                                                                      trs_merged_map[sid]["mpDebugInfo"] else -1
            # sort_rank 不同的打印
            if target in other_targets:
                rank_select_info[sid].append(trs_score)
                rank_select_info[sid].append(taf_score)
            elif not ((target == 'sort_rank' and trs_score == taf_score)
                      or (target.find("score") >= 0 and abs(float(trs_score) - float(taf_score)) / (
                            float(trs_score) + 1e-9) < 0.0)
                      or (target.find("post_selected_reason") >= 0 and trs_score == taf_score)):
                f.write("{},{} =|= {}\n".format(sid, trs_score, taf_score))
        f.write("\n")

    f.write("===============rsp dscore====================\n")
    for sid in both_rsp_id_set:
        taf_score = taf_merged_map[sid]["dScore"]
        trs_score = trs_merged_map[sid]["dScore"]

        f.write("{},{} =|= {}\n".format(sid, trs_score, taf_score))
    f.write("\n")

    rank_select_info_lst = [(sid, trs_rank, trs_select, int(taf_rank), taf_select) \
                            for sid, [trs_select, taf_select, trs_rank, taf_rank] in rank_select_info.items()]
    rank_select_info_lst.sort(key=lambda x: x[3])   # order by taf_rank
    f.write("===============merge rank target: post_rank_select:\n")
    for sid, trs_rank, trs_select, taf_rank, taf_select in rank_select_info_lst:
        f.write("{}: {}_{} =|= {}_{}\n".format(sid, trs_rank, trs_select, taf_rank, taf_select))

    post_targets = ["adjust_score", "adjust_detail_info"]
    for target in post_targets:
        f.write("===============recall merge target: {}\n".format(target))
        for sid in trs_recall_merged_id_set & taf_recall_merged_id_set:
            taf_score = taf_recall_merged_map[sid]["mpDebugInfo"][target] if "mpDebugInfo" in taf_recall_merged_map[
                sid] and target in taf_recall_merged_map[sid]["mpDebugInfo"] else -1
            trs_score = trs_recall_merged_map[sid]["mpDebugInfo"][target] if "mpDebugInfo" in trs_recall_merged_map[
                sid] and target in trs_recall_merged_map[sid]["mpDebugInfo"] else -1
            # adjust_score 且 不超过xx阈值，不打印
            if target == "recall_filter_reason" and taf_score == -1: taf_score = 0
            if not (target == 'adjust_score' and abs(float(trs_score) - float(taf_score)) <= 0.0 * float(trs_score)):
                f.write("{},trs-{} =|= taf-{}\n".format(sid, trs_score, taf_score))
        f.write("\n")

    f.close()

