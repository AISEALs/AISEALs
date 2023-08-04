#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2021tencent.com, Inc. All Rights Reserved
#
########################################################################

"""
File Name: a.py
Author: Wang Zhihua <wangzhihua@tencent.com>
Create Time: 2021/02/04 10:05:00
Brief:
"""

import sys
import os
import logging
import time
import json
import datetime

flag = ''
scene_dict = {}

fd = open('test-2897491-20210304112721.csv', 'r')
for line in fd:
    line = line.strip('\n')
    if 'cnt' in line:
        if 'favor' in line:
            flag = 'favor'
        if 'share' in line:
            flag = 'share'
        if 'account_level' in line:
            flag = 'account_level'
        continue
    cols = line.split('\t')
    cnt = int(cols[0])
    scene = cols[1]
    key = cols[2]
    vv_rank = cols[3]
    if scene not in scene_dict:
        scene_dict[scene] = {}
    if vv_rank not in scene_dict[scene]:
        scene_dict[scene][vv_rank] = {'total': 0, 'values': {}}
    scene_dict[scene][vv_rank]['total'] += cnt
    scene_dict[scene][vv_rank]['values'][key] = cnt

scenes = list(scene_dict.keys())
scenes = sorted(scenes)
print(flag)
for scene in scenes:
    print('=' * 20)
    print('scene: ' + scene)
    vvs = sorted(list(scene_dict[scene].keys()))
    total = 0
    for vv in vvs:
        scene_kv = scene_dict[scene][vv]
        scene_total = float(scene_kv['total'])
        total += scene_total
    for vv in vvs:
        print('-' * 20)
        print('vv_rank: ' + vv)
        scene_kv = scene_dict[scene][vv]
        scene_total = float(scene_kv['total'])
        # print('rate: ' + str(scene_total / float(total)))
        print(f'rate: {"%.2f" % (scene_total / float(total) * 100)}%')
        scene_values = scene_kv['values']
        keys = sorted(list(scene_values.keys()))
        for key in keys:
            # print('%s %i %f' % (key, scene_values[key], float(scene_values[key] / scene_total)))
            print(f'{key}, {scene_values[key]}, {"%.2f" % float(scene_values[key] / scene_total*100)}%')
    print('')

