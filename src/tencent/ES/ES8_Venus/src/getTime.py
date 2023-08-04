
import time

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


if __name__ == "__main__":
    print(nextHour("2021101323"))