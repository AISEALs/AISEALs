#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import codecs
import json
import argparse
import time
from itertools import islice


def readnlines(ifname, n):
    with codecs.open(ifname, 'r', encoding='utf-8') as ifile:
        return list(islice(ifile, 0, int(n)))


def write2f(ofname, data):
    with codecs.open(ofname, 'w', encoding='utf-8') as ofile:
        ofile.write(data + '\n')


def get_slot2fea(line):
    slot2feas = {}

    headfea = line.strip().split("|")
    features = headfea[7]

    fea_list = features.split(";")
    for fea in fea_list:
        ksv = fea.split(":")  # ksv for KeySlotVal
        if len(ksv) != 2 and len(ksv) != 3:
            print('skip invalid feature = ', fea)
            continue
        key = int(ksv[0])
        slot = int(ksv[1])
        val = float(ksv[2]) if len(ksv) == 3 else 1.0
        feas = slot2feas.get(slot, [])
        feas.append((key, val))
        slot2feas[slot] = feas

    return slot2feas


# convert origin format(key:slot:value) to trpc csr json
def convert_origin2csr(line):
    slot2feas = get_slot2fea(line)
    if slot2feas == None:
        return None
    sample = {}
    sample['slot_id'] = list(slot2feas.keys())
    sample['slot_id_offset'] = [0]
    sample['idx'] = []
    sample['val'] = {"float_list": {"value": []}}
    for slot in sample['slot_id']:
        kv_list = slot2feas[slot]
        sample['slot_id_offset'].append(sample['slot_id_offset'][-1] + len(kv_list))
        for k, v in kv_list:
            sample['idx'].append(k)
            sample['val']['float_list']['value'].append(v)
    return sample


def convert_origin2normal(line):
    slot2feas = get_slot2fea(line)
    if slot2feas == None:
        return None

    slots = []
    for slot, kv_list in slot2feas.items():
        features = [{'idx': k, 'val': v} for k, v in kv_list]
        slots.append({'slot_id': slot, 'features': features})
    sample = {"slots": slots, "sample_name": ""}
    return sample


def convert(origin_lines, args):
    inputs = []
    for line in origin_lines:
        if args.output_type == 'csr':
            sample = convert_origin2csr(line)
        elif args.output_type == 'normal':
            sample = convert_origin2normal(line)
        else:
            print('invalid args.output_type = ', args.output_type)

        if sample != None:
            inputs.append(sample)

    res = {}
    res['req_id'] = str(int(time.time()))
    res['inputs'] = inputs
    res['output_filter'] = args.output_filter.split(',')
    res['middle_inputs'] = {}
    res['custom_data'] = ''
    return json.dumps(res, ensure_ascii=False)


def parse_argument():
    parser = argparse.ArgumentParser(
        description="无量Serving的请求格式转换, 从训练样本格式到json-csr或json-normal的json格式",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", help="input file name", default='./part-00000')
    parser.add_argument("-n", "--num", help="number of lines to read from input file", default=1)
    parser.add_argument("-of", "--output_filter", help="output_filter, 多目标时用逗号分割", default='ctr')
    parser.add_argument("-otype", "--output_type", help="output格式, 只能是normal或者csr", default='csr')
    parser.add_argument("-o", "--output", help="output file name prefix", default='./output')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_argument()
    lines = readnlines(args.input, args.num)
    res = convert(lines, args)
    write2f(args.output + '-' + args.output_type + '.json', res)
