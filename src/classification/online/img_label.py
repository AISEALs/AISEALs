import io
import pycurl
from concurrent.futures import ThreadPoolExecutor
from online_img_model import *



def predictWithTrainedModel(test_imgs,today):
    predict_result = {}
    to_predict_imgs = []
    for i in test_imgs:
        if not os.path.exists(i):
            write_file("img :%s is not exists" % i,"log/log.wf")
        else:
            to_predict_imgs.append(i)


    try:
        tmp = requestDLServerLabels(to_predict_imgs)
        if tmp.features.feature.get("classesFood") == None:
            write_file("requestDLServerLabels is None","log/log.wf")
            return predict_result

        write_file(",".join(to_predict_imgs) + "\n" + str(tmp),'log/online_model.%s' % today)
        label1 = tmp.features.feature.get("classesFood").int64_list.value
        label2 = tmp.features.feature.get("classesPP").int64_list.value
        label3 = tmp.features.feature.get("classesScene").int64_list.value
        probs1 = tmp.features.feature.get("logitsFood").float_list.value
        probs2 = tmp.features.feature.get("logitsPP").float_list.value
        probs3 = tmp.features.feature.get("logitsScene").float_list.value


        if len(label1) != len(to_predict_imgs):
            write_file("img send to predict result nums:%s,img get back result nums:%s" % (len(to_predict_imgs),len(result)),"log/log.wf")
            return predict_result
        else:
            for i in range(0,len(label1)):      
                label_ids   = []
                label_names = []                
                if label1[i] == 1:              
                    label_ids.append("869")
                    label_names.append("晒美食小吃")       
                if label2[i] == 1:              
                    label_ids.append("872")     
                    label_names.append("个人自拍")
                if label3[i] == 1:              
                    label_ids.append("871")
                    label_names.append("晒风景/室内外环境")
                predict_result[to_predict_imgs[i]] = [label_ids,label_names] 

            return predict_result

    except Exception as err:
        write_file("Img predict err:%s" % err,"log/log.wf")
        return predict_result




'''def write_file(line,filename):
  file_obj = open(filename, 'a+')

  if filename.find("/log") != -1 or filename.find("timer") != -1:
    now_tm = time.localtime(int(time.time()))
    cur_time = time.strftime("%Y-%m-%-d %H:%M:%S", now_tm)
    line = "%s ----- %s" %  (cur_time,line)

  file_obj.write(line + "\n")
  file_obj.close()'''

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
  img_name = url.split("/")[-1].split(".")[0] + ".jpg"
  img_name = "img_data/%s" % img_name
  if not os.path.exists(img_name):
    result = download(url)
    if result:
      return True
    else:
      retry = 5
      while retry > 0 and not result:
        result = download(url)
        retry -= 1

      return result
  else:
      return True

def img_predict(imgs,today):
    img_task = []
    relation = {}
    for i in imgs:
        for j in i[1]:
            img_task.append(j)
            img_name = j.split("/")[-1].split(".")[0] + ".jpg"
            relation[img_name] = str(i[0])

    if len(img_task) == 0:
        return {},True

    '''download imgs'''
    start_1 = time.time()
    with ThreadPoolExecutor(24) as pool:
      result = pool.map(img_download, img_task)

    start_2 = time.time()
    write_file("%s img download time:%s" % (len(img_task),(start_2 - start_1)),"log/timer.%s" % today)
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
    if len(predict_imgs) == 0:
        return {},True

    start_3 = time.time()
    predict_result = predictWithTrainedModel(predict_imgs,today)
    start_4 = time.time()
    write_file("requestDLServerLabels time:%s" % (start_4 - start_3),"log/timer.%s" % today)
    if len(predict_result) == 0:
        return {},False

    img_face_label = {}
    for k,v in predict_result.items():
        if k.split("/")[1] not in relation:
            write_file("strange","log/log.wf")
            continue
        if len(v[0]) > 0:#"personal_photo":
            img_face_label[k.split("/")[1]] = v
            write_file(relation[k.split("/")[1]] + '\t' + k.split("/")[1] + '\t' + ",".join(v[0]),"log/img_predict.%s" % today)
        else:
            write_file(relation[k.split("/")[1]] + '\t' + k.split("/")[1] + '\tothers' ,"log/img_predict.%s" % today)

    img_predict_result = {}
    for i in imgs:
        finally_label_ids = []
        finally_label_names = []
        for j in i[1]:
            img_name = j.split("/")[-1].split(".")[0] + ".jpg"
            if img_name in img_face_label:
                for k in img_face_label[img_name][0]:
                  if k not in finally_label_ids:
                    finally_label_ids.append(k)
                for k in img_face_label[img_name][1]:
                  if k not in finally_label_names:
                    finally_label_names.append(k)

        if len(finally_label_ids) > 0:
          img_predict_result[i[0]] = [finally_label_ids,1,1,",".join(finally_label_names)]
        else:
          write_file(str(i[0]) + '\tothers' ,"log/log.%s" % today)

                #img_predict_result[i[0]] = ['872',img_face_label[img_name],img_face_label[img_name],'个人自拍']
                #break


    start_5 = time.time()
    write_file("finish img process time:%s" % (start_5 - start_4),"log/timer.%s" % today)
    return img_predict_result,True



