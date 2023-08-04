#!/usr/bin/env python
# -*- coding:utf-8 -*-

from taf.core import tafcore
from taf.dcache import *
import time
import random
import os
import sys

BETA_KEY = ["ES_MINIFLOAT_SKIP_BETA", "ES_MINIFLOAT_RATIO_BETA", "ES_MINIFLOAT_TIME_BETA", "ES_MINIFLOAT_ISFINISH_BETA",
"ES_MINIFLOAT_SKIP_ORDER_BETA", "ES_MINIFLOAT_RATIO_ORDER_BETA", "ES_MINIFLOAT_TIME_ORDER_BETA", "ES_MINIFLOAT_ISFINISH_ORDER_BETA"]
THETA_FILE = "data/all_theta_dict.txt"
PART = 2
model_subversion = 0

def wirte2dcache(key, value):
    prx = dcacheProxy()
    key = key.encode('utf-8')
    value = value.encode('utf-8')
    prx.locator("DCache.mtttjaProxyServer.ProxyObj@tcp -h 9.220.205.122 -p 30998")
    module = "MTTKBFunFloatInfo"
    ret = prx.setString(module, key, value)
    time.sleep(1)
    ret, data = prx.getString(module, key)
    print (ret, data)
    

if __name__ == "__main__":
    home_path = './'
    model_version = sys.argv[1]
    PART = int(sys.argv[2])
    model_subversion = int(sys.argv[3])
    d = {}  # key - score
    theta_file = os.path.join(home_path, THETA_FILE)
    with open(theta_file) as f:
        for line in f:
            seps = line.strip().split("\t")
            if len(seps) != 2:
                continue
            key = seps[0]
            score = seps[1]
            t = ""
            for item in BETA_KEY:
                if key.startswith(item):
                    t = item
                    break
            slot_key = key[len(t)+1:]
            if t == "":
                continue
            if t not in d:
                d[t] = []
            d[t].append(slot_key + ":" + score)

    d2 = {} # split to 3 parts
    for k, v in d.items():
        #for i in range(0, PART):
        key = k + "_" + str(PART)
        if key not in d2:
            d2[key] = []
        for i in range(0, len(v)):
            #j = i % 3
            #key = k + "_" + str(j)
            d2[key].append(v[i])
    for k, v in d2.items():
        print(k)
        print(model_version + "&" + "|".join(v))
        wirte2dcache(k, model_version + "&" + "|".join(v) + "&" + str(model_subversion))
