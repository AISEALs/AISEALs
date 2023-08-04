from  collections import defaultdict
import pandas as pd

trs = """
[2022-11-21 16:54:12.453] [thread 3281861] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: AccountFollowMiniVideoRecallServer | req:200 => rsp:0
[2022-11-21 16:54:12.453] [thread 3281861] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: GERecallServer | req:100 => rsp:0
[2022-11-21 16:54:12.455] [thread 3281861] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: ContinueRecallExternalOLASevenMINI | req:150 => rsp:0
[2022-11-21 16:54:12.455] [thread 3281861] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: GeneralI2IFloatRecallServer | req:50 => rsp:0
[2022-11-21 16:54:12.456] [thread 3281861] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: ContinueRecallExternalChooseMINI | req:150 => rsp:0
[2022-11-21 16:54:12.457] [thread 3281861] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: GroupRecallServer | req:200 => rsp:0
[2022-11-21 16:54:12.459] [thread 3281861] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: PaidColumnVideoCBRecallServer | req:50 => rsp:29
[2022-11-21 16:54:12.461] [thread 3281861] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: VectorVideoRecallServer | req:100 => rsp:0
[2022-11-21 16:54:12.463] [thread 3281861] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: float_graph_matching_server | req:100 => rsp:0
[2022-11-21 16:54:12.473] [thread 3281862] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: TRSItemCFFloatMatchingServer | req:531 => rsp:0
[2022-11-21 16:54:12.475] [thread 3281862] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: VideoFloatExploreServer | req:30 => rsp:30
[2022-11-21 16:54:12.481] [thread 3281862] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: SPVideoSemanticRecallServer | req:100 => rsp:100
[2022-11-21 16:54:12.483] [thread 3281863] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: HotMiniVideoCBRecallServer | req:100 => rsp:82
[2022-11-21 16:54:12.494] [thread 3281864] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: video_float_cb_recall_server | req:500 => rsp:498
[2022-11-21 16:54:12.497] [thread 3281861] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: TRSHPMNMatchingServer | req:100 => rsp:0
[2022-11-21 16:54:12.500] [thread 3281862] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: followTargetRecallServer | req:100 => rsp:26
[2022-11-21 16:54:12.513] [thread 3281863] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: TRSBertMatchingServer | req:128 => rsp:128
[2022-11-21 16:54:12.515] [thread 3281861] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: TRSFloatVideoSemanticMatchingServer | req:834 => rsp:457
[2022-11-21 16:54:12.529] [thread 3281864] [info] [mtt/preranking/core/ispine_ops/matching_op.cc:711] [ValidateRequestNum] debug_id: 146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430, server: TRsMiniVideoUCFMatchingServer | req:275 => rsp:275
"""
taf = """
11.185.224.221|2022-11-21 16:54:13|query_id:146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430|server:GroupRecallServer|req_size:200|count:0
11.185.224.221|2022-11-21 16:54:13|query_id:146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430|server:AccountFollowMiniVideoRecallServer|req_size:200|count:0
11.185.224.221|2022-11-21 16:54:13|query_id:146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430|server:GERecallServer|req_size:100|count:0
11.185.224.221|2022-11-21 16:54:13|query_id:146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430|server:ContinueRecallExternalChooseMINI|req_size:150|count:0
11.185.224.221|2022-11-21 16:54:13|query_id:146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430|server:ContinueRecallExternalOLASevenMINI|req_size:150|count:0
11.185.224.221|2022-11-21 16:54:13|query_id:146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430|server:PaidColumnVideoCBRecallServer|req_size:50|count:29
11.185.224.221|2022-11-21 16:54:13|query_id:146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430|server:SPVideoSemanticRecallServer|req_size:100|count:100
11.185.224.221|2022-11-21 16:54:13|query_id:146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430|server:HotMiniVideoCBRecallServer|req_size:100|count:83
11.185.224.221|2022-11-21 16:54:13|query_id:146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430|server:TRSItemCFFloatMatchingServer|req_size:531|count:0
11.185.224.221|2022-11-21 16:54:13|query_id:146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430|server:VectorVideoRecallServer|req_size:100|count:0
11.185.224.221|2022-11-21 16:54:13|query_id:146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430|server:float_graph_matching_server|req_size:100|count:0
11.185.224.221|2022-11-21 16:54:13|query_id:146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430|server:TRSBertMatchingServer|req_size:128|count:128
11.185.224.221|2022-11-21 16:54:13|query_id:146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430|server:followTargetRecallServer|req_size:100|count:0
11.185.224.221|2022-11-21 16:54:13|query_id:146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430|server:TRSFloatVideoSemanticMatchingServer|req_size:834|count:457
11.185.224.221|2022-11-21 16:54:13|query_id:146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430|server:TRSHPMNMatchingServer|req_size:100|count:100
11.185.224.221|2022-11-21 16:54:13|query_id:146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430|server:TRsMiniVideoUCFMatchingServer|req_size:275|count:275
11.185.224.221|2022-11-21 16:54:13|query_id:146cfd8ec1b46bfaa3f132d975e188cb_1667480418402_1669020852430|server:video_float_cb_recall_server|req_size:500|count:498
"""

server_num_map = defaultdict(lambda : [0, 0, 0, 0])
for line in trs.split('\n'):
    if not line:
        continue
    parts = line.split(' ')
    server = parts[10]
    req = int(parts[12].split(':')[1])
    rsp = int(parts[14].split(':')[1])
    server_num_map[server][0] += req
    server_num_map[server][1] += rsp

for line in taf.split('\n'):
    if not line:
        continue
    line = line.replace(',', '')
    parts = line.split('|')
    server = parts[3].split(':')[1]
    req = int(parts[4].split(':')[1])
    rsp = int(parts[5].split(':')[1])
    server_num_map[server][2] += req
    server_num_map[server][3] += rsp

df = pd.DataFrame.from_dict(server_num_map, orient='index', columns=['trs-req', 'trs-rsp', 'taf-req', 'taf-rsp']).sort_index()
df = df[(df['trs-rsp'] != 0) | (df['taf-rsp'] != 0)]
# df = pd.DataFrame.from_dict(server_num_map)

print(df)

