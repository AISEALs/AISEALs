from urllib.request import urlretrieve
import tensorflow as tf
import model_tool
import os
import io
import sys
import time
import csv
import pycurl
from PIL import Image
import urllib.request
from concurrent.futures import ThreadPoolExecutor


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
tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)

testImagePath = tf.placeholder(dtype=tf.string, name='dataInput')
testImgTensor = tf.image.decode_jpeg(tf.read_file(testImagePath), channels=3)
imagesExpanded = tf.expand_dims(testImgTensor, 0)
modelCls = model_tool.Model(is_training=False)
imgResized = modelCls.preprocess(imagesExpanded, FLAGS.resize, FLAGS.resize)

prediction_dict = modelCls.precision(imgResized)
postprocessed_dict = modelCls.postprocess(prediction_dict)
classes = postprocessed_dict['classes']
score = postprocessed_dict['logits']

def getScroe(clsClass, clsScore):
    for i in range(clsScore.shape[1]):
        if clsClass[0] == i:
            return clsScore[0][i]
    return None

def predictWithTrainedModel(trainedModelPath, test_imgs, today):
    result = {}
    try:
        with tf.Session() as sess:
            tf.train.Saver().restore(sess, trainedModelPath)
            #sess.graph.finalize()

            success_img = 0
            fail_img = 0
            predict_start_time = time.time()

            for img in test_imgs:
                try:
                    if not os.path.exists(img):
                        write_file("img :%s is not exists" % img,"log/log.wf")
                        fail_img += 1
                        continue
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
                    write_file("Img predict err:%s,img:%s" % (e,img),"log/log.wf")
                    continue


            predict_end_time = time.time()
            write_file("total imgage number:%s,fail predict image number:%s,success predict image number:%s" % (fail_img + success_img,fail_img,success_img),"log/log.%s" % today)
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
        write_file("Img predict err:%s" % err,"log/log.wf")
        return result




def write_file(line,filename):
  file_obj = open(filename, 'a+')

  if filename.find("/log") != -1:
    now_tm = time.localtime(int(time.time()))
    cur_time = time.strftime("%Y-%m-%-d %H:%M:%S", now_tm)
    line = "%s ----- %s" %  (cur_time,line)

  file_obj.write(line + "\n")
  file_obj.close()

def download(url,proxy=None,port=None):
  try:
    url = url + "?t=1"
    c = pycurl.Curl()
    b = io.BytesIO()
    if proxy != None and port != None:
      c.setopt(pycurl.PROXY,proxy)
      c.setopt(pycurl.PROXYPORT, int(port))

    c.setopt(pycurl.WRITEFUNCTION, b.write)
    if url.startswith("//"):
      c.setopt(pycurl.URL, "http:" + url)
    else:
      c.setopt(pycurl.URL, url)
    c.setopt(pycurl.CONNECTTIMEOUT,10)
    c.setopt(pycurl.TIMEOUT,10)
    c.setopt(pycurl.HEADER, 0)
    c.setopt(pycurl.USERAGENT, 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.79 Safari/537.1')
    c.perform()

    code = c.getinfo(c.HTTP_CODE)
    if code == 200:
        response = b.getvalue()
        img_name = url.split("/")[-1].split(".")[0] + ".jpg"
        f = open("img_data/%s" % img_name, 'wb')
        f.write(response)
        f.close()
        return True
    else:
        #write_file("%s fail,code:%s" % (url,code),'log/img.wf')
        return False

  except Exception as ex:
    #write_file("%s fail,ex:%s" % (url,ex),'log/download.wf')
    return False


def img_download(url):
  result = download(url)
  if result:
    return True
  else:
    retry = 5
    while retry > 0 and not result:
      result = download(url)
      retry -= 1

    return result


def img_predict(imgs,today):
    img_task = []
    relation = {}
    for i in imgs:
        for j in i[1]:
            img_task.append(j)
            img_name = j.split("/")[-1].split(".")[0] + ".jpg"
            relation[img_name] = str(i[0])

    '''download imgs'''
    with ThreadPoolExecutor(24) as pool:
      result = pool.map(img_download, img_task)

    j = 0
    predict_imgs = []
    success_download = 0

    for ret in result:
      if not ret :
          write_file("img:%s download 5 times and failed" % img_task[j],"log/log.wf")
          #下载失败视作非自拍
      else:
          success_download += 1
          img_name = img_task[j].split("/")[-1].split(".")[0] + ".jpg"
          predict_imgs.append("img_data/%s" % img_name)
      j += 1

    write_file("下载图片:%s张，成功%s张" % (j,success_download),"log/log.%s" % today)
    img_predict_result = {}
    if len(predict_imgs) == 0:
        return img_predict_result

    predict_result = predictWithTrainedModel(
                        trainedModelPath=FLAGS.trainedModel,
                        test_imgs=predict_imgs,
                        today=today
                        )

    img_face_label = {}
    for k,v in predict_result.items():
        if k.split("/")[1] not in relation:
            write_file("strange","log/log.wf")
            continue
        else:
            write_file(relation[k.split("/")[1]] + '\t' + k.split("/")[1] + '\t' + v[0] + '\t' + str(v[1]),"log/img_predict.%s" % today)
        if v[0] == "personal_photo":
            img_face_label[k.split("/")[1]] = v[1]

    img_predict_result = {}
    for i in imgs:
        for j in i[1]:
            img_name = j.split("/")[-1].split(".")[0] + ".jpg"
            if img_name in img_face_label:
                img_predict_result[i[0]] = ['872',img_face_label[img_name],img_face_label[img_name],'个人自拍']
                break



    return img_predict_result



