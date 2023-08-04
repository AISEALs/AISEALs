import sys
import os

cur_dirname = os.path.dirname(os.path.abspath(__file__))
src_dirname = os.path.abspath(os.path.join(cur_dirname, ".."))
root_dirname = os.path.abspath(os.path.join(cur_dirname, "../.."))

sys.path.append(os.path.join(root_dirname, "src"))

DATABASE_CONFIG = {
    'host': 'localhost',
    'dbname': 'company',
    'user': 'user',
    'password': 'password',
    'port': 3306
}

import socket
hostname = socket.gethostname()
try:
    myname = socket.getfqdn(hostname)
    myaddr = socket.gethostbyname(myname)
except:
    myaddr = '127.0.0.1'
    # myaddr = socket.gethostbyname(hostname)
    pass
# 测试服务器部署地址
test_server_addr = "127.0.0.1"
local_host = "127.0.0.1"
debug_mode = myaddr == test_server_addr
# debug_mode = False
local_host_mode = myaddr == local_host

log_path = os.path.join(root_dirname, "log")
if not os.path.exists(log_path):
    os.mkdir(log_path)

from src.tools.logger import get_logger
logger = get_logger(os.path.join(log_path, "log.base"))

logger.info("-" * 40)
logger.info("NOTICE test server addr:[{}]\nnow server addr:[{}]".format(test_server_addr, myaddr))
logger.info("NOTICE now debug mode:[{}] local host mode:[{}]".format(debug_mode, local_host_mode))
logger.info("root dir: {}".format(root_dirname))
logger.info(f"src dir: {src_dirname}")
logger.info("-" * 40)
