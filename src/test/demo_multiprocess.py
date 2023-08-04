import multiprocessing
import time


def get_result(num):
    process_name = multiprocessing.current_process().name
    print("Current process:", process_name, ", Input Number:", num)
    time.sleep(5)
    return 10*num


if __name__ == '__main__':
    number = [2, 4, 6, 8]
    with multiprocessing.Pool(2) as pool:
        # mylist = pool.map(func=get_result, iterable=number)
        # mylist = [pool.apply(func=get_result, args=(num,)) for num in number]
        mylist = [pool.apply_async(func=get_result, args=(num,)) for num in number]
        mylist = [p.get() for p in mylist]
        # pool.close()
        # pool.join()
        print("Output:", mylist)