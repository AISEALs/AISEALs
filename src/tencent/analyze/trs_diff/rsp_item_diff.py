import json
import sys
from collections import defaultdict


def dict_generator(indict, pre=None):
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                if len(value) == 0:
                    yield pre + [key, '{}']
                else:
                    for d in dict_generator(value, pre + [key]):
                        yield d
            elif isinstance(value, list):
                if len(value) == 0:
                    yield pre + [key, '[]']
                elif isinstance(value[0], float | int):
                    yield pre + [key, ', '.join(list(map(str, value)))]
                else:
                    for v in value:
                        for d in dict_generator(v, pre + [key]):
                            yield d
            elif isinstance(value, tuple):
                if len(value) == 0:
                    yield pre + [key, '()']
                else:
                    for v in value:
                        for d in dict_generator(v, pre + [key]):
                            yield d
            else:
                yield pre + [key, value]
    else:
        yield indict


if __name__ == "__main__":
    # output = open("data/result/result.diff", "w")
    output = sys.stdout

    query_id = "1ff70fcf9a74dc87cb4bbfbf1b1188cb_1669628853098_1669894974575"
    taf_final_rsp_items = {}
    # taf_final_rsp_items = defaultdict(list)
    with open(f"./data/{query_id}/taf_rsp") as f:
        taf_json = json.load(f)
        for item in taf_json['vData']:
            if 'returnFinalRsp' in item['mpExtInfo']:
                taf_final_rsp_items[item['lId']] = item
            # taf_final_rsp_items[item['lId']].append(item)

    trs_final_rsp_items = {}
    with open(f"./data/{query_id}/trs_rsp") as f:
        trs_json = json.load(f)
        for item in trs_json['vData']:
            # print(item['mpExtInfo'])
            if 'returnFinalRsp' in item['mpExtInfo']:
                trs_final_rsp_items[item['lId']] = item

    # if item_id not in trs_final_rsp_items or item_id not in taf_final_rsp_items:
    keys = set(trs_final_rsp_items.keys() & taf_final_rsp_items.keys())
    print(f'taf rsp size:{len(taf_final_rsp_items)}, trs rsp size:{len(trs_final_rsp_items)}, both size:{len(keys)}')
    item_id = list(keys)[1]

    for item_id in keys:
        output.write(f"item_id: {item_id}\n")
        trs_dict = {}
        trs_str = trs_final_rsp_items[item_id]
        for i in dict_generator(trs_str):
            try:
                value = '.'.join(list(map(str, i)))
                trs_dict[value] = 1
            except:
                pass

        taf_str = taf_final_rsp_items[item_id]
        for i in dict_generator(taf_str):
            kv = '.'.join(list(map(str, i)))
            if "scale" in kv:
                print(kv)
            if kv not in trs_dict:
                k = '.'.join(list(map(str, i))[:-1])
                trs_vals = []
                for k2, _ in trs_dict.items():
                    if k2.startswith(k + '.'):
                        trs_vals.append(k2)
                if len(trs_vals) > 0:
                    output.write(f"taf:[{kv}] =!= trs:[{'.'.join(trs_vals)}]\n")
                else:
                    output.write(f"key:[{kv}] not be found in trs\n")

            # elif trs_dict[key] != value:
            #     output.write(f"key:[{key}], taf value:{value} != trs value:{trs_dict[key]}\n")
            else:
                # output.write(f"key:[{kv}] OK!\n")
                pass
        break

    output.close()
