import json
import sys
import traceback


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
    # output = open("./data/result/result.diff", "w")
    output = sys.stdout

    with open("data/result/taf.json") as f:
        taf_json = json.load(f)
        # print(taf_json)
        valid_keys = ['reqInfo', 'userProfile', 'exposeData']
        del_keys = [k for k in taf_json.keys() if k not in valid_keys]
        for k in del_keys:
            del taf_json[k]

    with open("data/result/trs.json") as f:
        trs_json = json.load(f)

    trs_dict = {}
    for i in dict_generator(trs_json):
        # key = '.'.join(i[0:-1])
        # value = i[-1]
        # trs_dict[key] = value
        try:
            value = '.'.join(list(map(str, i)))
            trs_dict[value] = 1
        except:
            pass

    for i in dict_generator(taf_json):
        kv = '.'.join(list(map(str, i)))
        # key = '.'.join(i[0:-1])
        # value = i[-1]
        # if kv.startswith("userProfile.userProfilePb.userModel.mpInterests"):
        #     kv = kv.replace("interestItemList.id", "interestItemList.sid")
        if kv not in trs_dict:
            output.write(f"key:[{kv}] not be found in trs\n")
        # elif trs_dict[key] != value:
        #     output.write(f"key:[{key}], taf value:{value} != trs value:{trs_dict[key]}\n")
        else:
            # output.write(f"key:[{kv}] OK!\n")
            pass

    output.close()
