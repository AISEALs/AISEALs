import os
import sys
from time import strftime, localtime, sleep
import logging
from getTime import nextHour

HADOOP = "/data/home/carliu/hadoop-mdfs/bin/hadoop"
MDFS_PATH = "mdfs://cloudhdfs/pcfeeds/qbdata/nemoztwang/MiniVideo/Float/Praise/es"
HOME_PATH = "/data/home/nemoztwang/ES"
MODEL_VERSION = "11"


def is_mdfs_exist(date_path):
    cmd_line = "{} fs -test -e {}/{}/part-00000".format(HADOOP, MDFS_PATH, date_path)
    ret = os.popen(cmd_line)
    logging.debug(ret.read())
    ret.close()
    return 0


def get_mdfs_file(date_path):
    cmd_line = "{} fs -get {}/{}/part-00000 {}/data/one_step_tmp/part-00000".format(HADOOP,
                                                                                    MDFS_PATH, date_path, HOME_PATH)
    ret = os.popen(cmd_line)
    ret_info = ret.read()
    ret.close()
    logging.debug(ret_info)
    flag = "{}/part-00000': No such file or directory".format(date_path)
    if flag in ret_info:
        return False
    return True


def train_once(date_path):
    logging.debug("######################{}######################".format(date_path))
    ret = is_mdfs_exist(date_path)
    if ret:
        logging.debug("ERROR: {}/part-00000 not exist".format(date_path))
        return
    else:
        logging.debug("SUCC: {}/part-00000 exist".format(date_path))

    ret = get_mdfs_file(date_path)
    if ret:
        logging.debug("ERROR: {}/part-00000 not download".format(date_path))
        return
    else:
        logging.debug("SUCC: {}/part-00000 download".format(date_path))

    ret = os.system("python {}/src/pre_process.py {} {}".format(HOME_PATH, HOME_PATH, MODEL_VERSION))
    if ret:
        logging.debug("ERROR: pre_process")
        return
    else:
        logging.debug("SUCC: pre_process")

    ret = os.system("rm {}/data/one_step_tmp/part-00000".format(HOME_PATH))
    if ret:
        logging.debug("ERROR: rm {}/part-00000".format(date_path))
        return
    else:
        logging.debug("SUCC: rm {}/part-00000".format(date_path))

    ret = os.system("python {}/src/theta_trans.py {}".format(HOME_PATH, HOME_PATH))
    if ret:
        logging.debug("ERROR: theta_trans")
        return
    else:
        logging.debug("SUCC: theta_trans")

    ret = os.system("python {}/src/one_step_run.py {}".format(HOME_PATH, HOME_PATH))
    if ret:
        logging.debug("ERROR: one_step_run")
        return
    else:
        logging.debug("SUCC: one_step_run")

    ret = os.system("python {}/src/theta_trans2.py {}".format(HOME_PATH, HOME_PATH))
    if ret:
        logging.debug("ERROR: theta_trans2")
        return
    else:
        logging.debug("SUCC: theta_trans2")

    ret = os.system("mv {}/data/all_theta_dict.txt {}/data/all_theta_dict.txt.bak".format(HOME_PATH, HOME_PATH))
    if ret:
        logging.debug("ERROR: mv1")
        return
    else:
        logging.debug("SUCC: mv1")

    ret = os.system("mv {}/data/one_step_tmp/all_theta_dict.txt {}/data/all_theta_dict.txt".format(HOME_PATH, HOME_PATH))
    if ret:
        logging.debug("ERROR: mv2")
        return
    else:
        logging.debug("SUCC: mv2")

    ret = os.system("python {}/dcache/dcache_client.py {} {}".format(HOME_PATH, HOME_PATH, MODEL_VERSION))
    if ret:
        logging.debug("ERROR: dcache_client")
        return
    else:
        logging.debug("SUCC: dcache_client")



if __name__ == "__main__":
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename="train.log", filemode="a", level="DEBUG", format=LOG_FORMAT)
    # date_str = sys.argv[1]
    # date_path = "{}-{}-{}/{}".format(date_str[0:4], date_str[4:6], date_str[6:8], date_str[8:10])

    start_time = "2021111114"
    end_time = "2026101317"
    while True:
        start_time, date_path = nextHour(start_time)
        if start_time == end_time:
            break
        if int(start_time[8:10]) < 6:
            continue
        train_once(date_path)
        sleep(5*60)


