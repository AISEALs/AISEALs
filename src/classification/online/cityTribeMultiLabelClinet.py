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
import sys
sys.path.append("../")
import grpc
from protobuf import dlserver_pb2
from protobuf import dlserver_pb2_grpc
from protobuf import pb2_Feature, pb2_Features, pb2_Int64List, pb2_Example, pb2_BytesList
from conf import config

import cv2
import numpy as np


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
            # image_reader = ImageReader()
            # with tf.Session('') as sess:
            image = cv2.imread(imageName)  # 读入图片W
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图片转换为RGB格式，因为cv2读入为BGRW
            height = 299
            width = 299
            image = cv2.resize(image, (height, width))
            imageByteString = getImageByteString(image)

            #多标签W
            # imageName = getImageName(imageName)  # 获取图像对应的名字，用于预测W
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
    feature = stub.GetOutputs(dl_request)
    return feature


# host_port = '10.126.106.160:50051' if config.debug_mode else '10.126.106.210:50051'
host_port = '10.126.106.210:50051'
channel = grpc.insecure_channel(host_port)
stub = dlserver_pb2_grpc.ExamplesDLServerStub(channel)

def requestDLServerLabels(images):
    print("-------------- GetScores --------------")
    example_list = create_examples(images) #自定义解码W
    dl_request = dlserver_pb2.DLRequest(task_id="803", example_list=example_list) #服务idW
    return getScores(stub, dl_request)

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

    images = ['./tmp/image/WechatIMG{}.jpeg'.format(i) for i in range(17, 21)]

    for i in range(1):
        result = requestDLServerLabels(images)
        print(result)
