from urllib.request import urlretrieve
import tensorflow as tf
import model_tool
import os
import sys
import time
import csv
from PIL import Image
import urllib.request


slim = tf.contrib.slim
flags = tf.app.flags


flags.DEFINE_string('trainedModel',
                    './model/model.ckpt-27855',
                    'Path to trained model.')

flags.DEFINE_integer('resize',
                    299,
                    'The size of input images.')

flags.DEFINE_integer('classNum',
                    2,
                    'The number of classes.')

FLAGS = flags.FLAGS
labelMap = {0: 'others', 1: 'personal_photo'}


def getScroe(clsClass, clsScore):
    for i in range(clsScore.shape[1]):
        if clsClass[0] == i:
            return clsScore[0][i]
    return None

def predictWithTrainedModel(trainedModelPath, outputImgSize,test_imgs):
    result = {}
    try:
        testImagePath = tf.placeholder(dtype=tf.string, name='dataInput')
        testImgTensor = tf.image.decode_jpeg(tf.read_file(testImagePath), channels=3)
        imagesExpanded = tf.expand_dims(testImgTensor, 0)
        modelCls = model_tool.Model(is_training=False)
        imgResized = modelCls.preprocess(imagesExpanded, outputImgSize, outputImgSize)

        prediction_dict = modelCls.precision(imgResized)
        postprocessed_dict = modelCls.postprocess(prediction_dict)
        classes = postprocessed_dict['classes']
        score = postprocessed_dict['logits']

        with tf.Session() as sess:
            tf.train.Saver().restore(sess, trainedModelPath)
            sess.graph.finalize()

            success_img = 0
            fail_img = 0
            predict_start_time = time.time()

            for img in test_imgs:
                try:
                    imageInput, clsClass, clsScore = sess.run(\
                            [testImgTensor, classes, score],
                            feed_dict={testImagePath: img})

                    info = [img, round(clsScore[0][0], 4), round(clsScore[0][1], 4)]
                    success_img += 1
                    result[img] = [labelMap[clsClass[0]],\
                                   round(getScroe(clsClass,clsScore), 3)]

                except Exception as e:
                    tf.logging.error('something error happened : {} with test image :{}'.format(e, img))
                    fail_img += 1
                    continue


            predict_end_time = time.time()
            tf.logging.info(' predict time : {}, \n\
                              total imgage number:{},\n\
                              fail predict image number:{},\n\
                              success predict image number:{}'\
                              .format(
                              str(predict_end_time - predict_start_time) + 's',\
                              fail_img + success_img,\
                              fail_img,success_img,
                              ))
            return result
    except Exception as err:
        tf.logging.error(err)
        return result




tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)

#if __name__ == "__main__":
#  imgs = ["img_data/n_v2ffe912717166418e9cbd69c86c0ea3a4.jpg"]

def test(imgs):
  predict_result = predictWithTrainedModel(
                         trainedModelPath=FLAGS.trainedModel,
                         outputImgSize=FLAGS.resize,
                         test_imgs=imgs
                         )


  for k,v in predict_result.items():
      print(k,v)
