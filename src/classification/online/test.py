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

import sys
import argparse
import pprint

import grpc

import sys
sys.path.append("../")
from protobuf import dlserver_pb2
from protobuf import dlserver_pb2_grpc
import os
import cv2


def create_examples(images):
    import tensorflow as tf
    from protobuf import pb2_Feature, pb2_Features, pb2_Int64List, pb2_Example, pb2_BytesList

    class ImageReader(object):
        """Helper class that provides TensorFlow image coding utilities."""

        def __init__(self):
            # Initializes function that decodes RGB JPEG data.
            self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
            self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

        def read_image_dims(self, sess, image_data):
            image = self.decode_jpeg(sess, image_data)
            return image.shape[0], image.shape[1]

        def decode_jpeg(self, sess, image_data):
            image = sess.run(self._decode_jpeg,
                             feed_dict={self._decode_jpeg_data: image_data})
            assert len(image.shape) == 3
            assert image.shape[2] == 3
            return image

    def getImageName(inputStr):
        return os.path.split(inputStr)[-1]

    def getImageByteString(imgInput):
        try:
            image = tf.placeholder(dtype=tf.uint8)
            imageByteString = tf.image.encode_jpeg(image)
            with tf.Session() as sess:
                imageByteString = sess.run(imageByteString,feed_dict={image:imgInput})
            return imageByteString
        except Exception:
            return None

    def getExample(imageName):
        try:
            image_reader = ImageReader()
            # with tf.Session('') as sess:
            image = cv2.imread(imageName)  # 读入图片W
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图片转换为RGB格式，因为cv2读入为BGRW
            height = 299
            width = 299
            image = cv2.resize(image, (height, width))
            imageByteString = getImageByteString(image)


            # image_data = tf.gfile.FastGFile(imageName, 'rb').read()
            # height, width = image_reader.read_image_dims(sess, image_data)  # 获取图片的宽和高W
            imageName = getImageName(imageName)  # 获取图像对应的名字，用于预测W
            example = pb2_Example(features=pb2_Features(feature={
                'image/encoded': pb2_Feature(bytes_list=pb2_BytesList(value=[imageByteString])),
                'image/format': pb2_Feature(bytes_list=pb2_BytesList(value=[b'jpg'])),
                'image/class/label': pb2_Feature(int64_list=pb2_Int64List(value=[0])),
                'image/height': pb2_Feature(int64_list=pb2_Int64List(value=[height])),
                'image/width': pb2_Feature(int64_list=pb2_Int64List(value=[width])),
                'image/name': pb2_Feature(bytes_list=pb2_BytesList(value=[bytes(imageName, encoding='utf-8')])),
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
channel = grpc.insecure_channel('10.126.106.210:50051')
stub = dlserver_pb2_grpc.ExamplesDLServerStub(channel)

def requestDLServerLabels(images):
    #print("-------------- GetScores --------------")
    example_list = create_examples(images) #自定义解码W
    dl_request = dlserver_pb2.DLRequest(task_id="702", example_list=example_list) #服务idW
    return getScores(stub, dl_request)


def write_file(line,filename):
  file_obj = open(filename, 'a+')

  if filename.find("/log") != -1:
    now_tm = time.localtime(int(time.time()))
    cur_time = time.strftime("%Y-%m-%-d %H:%M:%S", now_tm)
    line = "%s ----- %s" %  (cur_time,line)

  file_obj.write(line + "\n")
  file_obj.close()


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
    write_file(str(tmp),'a')
    #tmp = result.features.feature
    probs =  tmp.features.feature.get("logits").float_list.value
    result = tmp.features.feature.get("classes").int64_list.value
    for i in range(0,len(result)):
        print(images[i],result[i],str(round(probs[i*2+1],3)),str(round(1-probs[i*2+1],3)))



