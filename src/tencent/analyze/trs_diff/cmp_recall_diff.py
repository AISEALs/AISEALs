import json
from collections import defaultdict
import pandas as pd


file1 = "data/trs_rsp"
file2 = "data/trs_rsp2"

with open(file1, "r", encoding='UTF-8') as trs_file:
    items1 = json.load(trs_file)

with open(file2, "r", encoding='UTF-8') as trs_file:
    items2 = json.load(trs_file)


server2ids1 = defaultdict(set)
for item in items1["vData"]:
    if "returnAllRecall" in item["mpExtInfo"]:
        recall_server = item['mpDebugInfo']['recall_server']
        server2ids1[recall_server].add(item['lId'])

server2ids2 = defaultdict(set)
for item in items2["vData"]:
    if "returnAllRecall" in item["mpExtInfo"]:
        recall_server = item['mpDebugInfo']['recall_server']
        server2ids2[recall_server].add(item['lId'])


server2list = defaultdict(list)
for server, item_ids1 in server2ids1.items():
    item_ids2 = server2ids2[server]
    # print(f'{server}, ids1 size:{len(item_ids1)}, ids2 size:{len(item_ids2)}, both size:{len(item_ids1&item_ids2)}')
    server2list[server].append(len(item_ids1))
    server2list[server].append(len(item_ids2))
    server2list[server].append(len(item_ids1 & item_ids2))

df = pd.DataFrame.from_dict(server2list, orient='index', columns=['query1size', 'query2size', 'bothsize']).sort_index()
print(df)


