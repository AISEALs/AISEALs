#!/usr/bin/python
#encoding: utf-8
import os
import pycurl
import io
import time
from concurrent.futures import ThreadPoolExecutor
from IMG import *

def write_file(line,filename):
  print(line)
  file_obj = open(filename, 'a+')

  if filename.find("log") != -1:
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
        write_file("%s fail,code:%s" % (url,code),'log/img.wf')
        return False

  except Exception as ex:
    write_file("%s fail,ex:%s" % (url,ex),'log/download.wf')
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


if __name__ == "__main__":
    img_task = ['https://pic8.58cdn.com.cn/mobile/big/n_v2ffe912717166418e9cbd69c86c0ea3a4.jpg','https://pic8.58cdn.com.cn/mobile/big/n_v2ffe915136448467284a3b38ce073d48d.jpg','https://pic8.58cdn.com.cn/mobile/big/n_v2ffe960a72b214f73b1c648cb0638d2c9.jpg','https://pic8.58cdn.com.cn/mobile/big/n_v2ffec7f9b15354213a98c956f7c8179de.jpg','https://pic8.58cdn.com.cn/mobile/big/n_v2ffeeca8ff3664baa8488cccaffedde1d.jpg','https://pic8.58cdn.com.cn/mobile/big/n_v2ffefc3855e3e46afacab7f3dd3178e6d.jpg','https://pic8.58cdn.com.cn/mobile/big/n_v2fff1b21ed6a24aab829303a46dd7bf1b.jpg','https://pic8.58cdn.com.cn/mobile/big/n_v2fff5cc67509d4e93b72416c74885893d.jpg','https://pic8.58cdn.com.cn/mobile/big/n_v2ffff521e80dc4f2cbb47b0b79bc07cae.jpg','https://pic8.58cdn.com.cn/mobile/big/n_v2ffff5aa8f99d4789bb2dd119696691fb.jpg']

    img2article = ['1','1','1','2','2','2','3','4','5','5']
    '''download imgs'''
    with ThreadPoolExecutor(24) as pool:
      result = pool.map(img_download, img_task)

    j = 0
    for ret in result:
      if not ret :
        write_file("img:%s of article:%s download 5 times and failed" % (img_task[j],img2article[j]),"log/log.wf")
      j += 1
