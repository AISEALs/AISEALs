# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the gRPC route guide client."""

from __future__ import print_function

import argparse
import pprint

import grpc
import sys
sys.path.append("../")
from protobuf import dlserver_pb2
from protobuf import dlserver_pb2_grpc
from protobuf import pb2_Feature, pb2_Features, pb2_Int64List, pb2_Example, pb2_BytesList
from conf import config

import os
import cv2
import time
import numpy as np

def write_file(line,filename):
  file_obj = open(filename, 'a+')

  if filename.find("/log") != -1 or filename.find("timer") != -1:
    now_tm = time.localtime(int(time.time()))
    cur_time = time.strftime("%Y-%m-%-d %H:%M:%S", now_tm)
    line = "%s ----- %s" %  (cur_time,line)

  file_obj.write(line + "\n")
  file_obj.close()

def create_examples(images):

    def getImageByteString(imgInput):
        try:
            img_encode = cv2.imencode('.jpg', imgInput)[1]
            data_encode = np.array(img_encode)
            imageByteString = data_encode.tostring()
            return imageByteString
        except Exception:
            return None

    def getExample(imageName):
        try:
            image = cv2.imread(imageName)  # 读入图片W
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图片转换为RGB格式，因为cv2读入为BGRW
            height = 299
            width = 299
            image = cv2.resize(image, (height, width))
            imageByteString = getImageByteString(image)
            example = pb2_Example(features=pb2_Features(feature={
                'image/encoded': pb2_Feature(bytes_list=pb2_BytesList(value=[imageByteString])),
                'image/format': pb2_Feature(bytes_list=pb2_BytesList(value=[b'jpg'])),
                'image/class/labelPP': pb2_Feature(int64_list=pb2_Int64List(value=[0])),
                'image/class/labelFood': pb2_Feature(int64_list=pb2_Int64List(value=[0])),
                'image/class/labelScene': pb2_Feature(int64_list=pb2_Int64List(value=[0]))
            }))
            return example
        except Exception as e:
            pass
        return None

    examples = list(filter(lambda x: x != None, map(getExample, images))) #将lines中的每一行转为TFRecordW

    print("request examples len: {}".format(len(examples)))
    example_list = dlserver_pb2.ExampleList(examples=examples) #生成一个list请求W

    return example_list


def getScores(stub, dl_request):
    # print(dl_request)
    feature = stub.GetOutputs(dl_request)
    # print(feature)
    return feature

# NOTE(gRPC Python Team): .close() is possible on a channel and should be
# used in circumstances in which the with statement does not fit the needs
# of the code.
# with grpc.insecure_channel('localhost:50051') as channel:
# with grpc.insecure_channel('127.0.0.1:50051') as channel:
host_port = '10.126.106.160:50051' if config.debug_mode else '10.126.106.210:50051'
channel = grpc.insecure_channel(host_port)
stub = dlserver_pb2_grpc.ExamplesDLServerStub(channel)

def requestDLServerLabels(images):
    #print("-------------- GetScores --------------")
    start_1 = time.time()
    example_list = create_examples(images) #自定义解码W
    start_2 = time.time()
    dl_request = dlserver_pb2.DLRequest(task_id="803", example_list=example_list) #服务idW
    result = getScores(stub, dl_request)
    start_3 = time.time()
    today = time.strftime("%Y%m%d",time.localtime(time.time()))
    write_file("create examples time:%s dl request time:%s" % ((start_2 - start_1), (start_3 - start_2)),"log/timer.%s" % today)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--debug_mode",
        type="bool",
        default=True,
        help="use debug mode")
    parser.add_argument(
        "--use_hdfs",
        type=bool,
        default=False,
        help="use hdfs file")
    parser.add_argument(
        "--partition_num",
        type=int,
        default=64,
        help="the number of the repartition num")

    FLAGS, unparsed = parser.parse_known_args()
    pp = pprint.PrettyPrinter().pprint
    pp(FLAGS)
    pp(unparsed)
    print("--------------------------")
    images = ['./tmp/image/WechatIMG{}.jpeg'.format(i) for i in range(17, 21)]
    tmp = requestDLServerLabels(images)
    if tmp.features.feature == None:
        print("a")
    #tmp = result.features.feature
    probs =  tmp.features.feature.get("logits").float_list.value
    result = tmp.features.feature.get("classes").int64_list.value
    for i in range(0,len(result)):
        print(images[i],result[i],str(round(probs[i*2+1],3)),str(round(1-probs[i*2+1],3)))



