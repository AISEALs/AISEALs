import os
import time
from datetime import datetime

def get_time():
    start_time = "2021101317"
    end_time = "2021101517"
    while True:
        start_t = time.strptime(start_time, "%Y%m%d%H")
        next_t_s = time.mktime(start_t) + 60 * 60

        next_t = time.localtime(next_t_s)
        next_t_h = time.strftime("%Y-%m-%d/%H", next_t)
        start_time = time.strftime("%Y%m%d%H", next_t)
        print(next_t_h)
        if start_time == end_time:
            break


def nextHour(start_time):
    start_t = time.strptime(start_time, "%Y%m%d%H")
    next_t_s = time.mktime(start_t) + 60 * 60

    next_t = time.localtime(next_t_s)
    next_t_h = time.strftime("%Y-%m-%d/%H", next_t)
    next_time = time.strftime("%Y%m%d%H", next_t)
    return next_time, next_t_h


def alarm(ts):
    train_data_time = time.strptime(ts, "%Y%m%d%H")
    train_time = time.strftime("%Y-%m-%d:%H", train_data_time)
    now_time = time.strftime('%Y-%m-%d:%H', time.localtime(time.time()))
    tt = time.mktime(train_data_time)
    nt = time.time()
    diff = nt - tt
    if 1 <= train_data_time.tm_hour <= 6:
	return
    print(nt, tt, diff)
    print(train_time, now_time)

    if diff > 60 * 60 * 3:
        os.system("sh /data/home/yimshi/send.sh " + "ES5_train_delay?train_data_time:{}now_time:{}".format(train_time, now_time) + " " + "@all")


if __name__ == "__main__":
    # print(nextHour("2021101323"))
    alarm("2021111823")
