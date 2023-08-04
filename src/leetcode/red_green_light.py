import time

lights = [('red', 5), ('yellow', 3), ('green', 8)]


## 红绿交通灯切换程序 #实现红绿灯的正常定时切换，绿灯8秒，红灯5秒，黄灯 3秒，红绿灯切换间都有黄灯，可以从红灯开始。
def watch_light(bias):
    bias = bias % 3

    cur = bias
    last = bias - 1

    while True:
        print(lights[cur][0])
        time.sleep(lights[cur][1])
        if cur > last:
            cur = cur + 1
            last = last + 1
            if cur == 3:
                cur = 1
        else:
            cur = cur - 1
            last = last - 1
            if cur == -1:
                cur = 1


if __name__ == '__main__':
    watch_light(bias=1)
