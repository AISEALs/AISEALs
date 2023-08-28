#!/usr/bin/python
# coding=utf-8
'''
函数处理
'''


def parseSlotsList(slot_config, to_str=False):
    # slot_config:'2,3,6-10,13-17' 左闭右闭
    if (slot_config == ""):
        return []
    configStr = slot_config.strip('[](){},')
    parts = configStr.split(',')
    reslist = []
    last = -1
    for p in parts:
        if p.find('-') > 0:
            pair = p.split('-')
            if len(pair) != 2:
                # print('can not parse:', p)
                return []
            left = int(pair[0])
            right = int(pair[1])
            if left <= last:
                # print('left <= last :', p)
                return []
            if right < left:
                # print('right < left :', p)
                return []

            for i in range(left, right + 1):
                if to_str:
                    reslist.append(str(i))
                else:
                    reslist.append(i)
            last = right
        else:
            cur = int(p)
            if cur <= last:
                # print('current <= last :', p)
                return []
            if to_str:
                reslist.append(str(cur))
            else:
                reslist.append(cur)
            last = cur

    return reslist


# 解析配置字符串
def parse_duration_config(config_str):
    boundaries = []
    weights = []
    ranges = config_str.split('|')
    for r in ranges:
        range_str, weight_str = r.split(':')
        if ',' in range_str:
            for val in range_str.split(','):
                boundaries.append(int(val))
        elif '-' in range_str:
            start, end = map(int, range_str.split('-'))
            boundaries.append(start)
            # 不要包括结束点，因为'-'代表左闭右开
            # 在这个例子中，我们不需要存储结束点，因为Bucketize函数也是左闭右开的
        weights.append(float(weight_str))
    return boundaries, weights


