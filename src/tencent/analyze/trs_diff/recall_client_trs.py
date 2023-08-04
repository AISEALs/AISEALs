#coding=utf-8
import pandas as pd
import xml.etree.ElementTree as ET

class Node:
    def __init__(self):
        self.server = None
        self.proto = "taf"
        self.timeout = 100
        self.obj = None
        self.req_num = 100

    def __str__(self):
        return f"""- server: {self.server}\n  obj: {self.obj}\n  proto: {self.proto}\n  timeout: {self.timeout}\n  req_num: {self.req_num}"""

"""
    读入XML数据，返回pa.DataFrame
"""
def read_xml(xml_FileName):
    with open(xml_FileName, "r") as xml_file:
        # 读取数据，以树的结构存储
        tree = ET.parse(xml_file)
        # 访问树的梗节点
        root = tree.getroot()
        # 返回DataFrame格式数据
        # return pd.DataFrame(list(iter_records(root)))
        return list(iter_records(root))

"""
    遍历有记录的生成器
"""
def iter_records(records):
    for record in records:
        # 保存值的临时字典
        # 遍历所有字段
        node = Node()
        text = record.text.strip()
        for attr in text.split("\n"):
            attr = attr.strip()
            if attr.startswith("obj="):
                node.obj = attr.replace("obj=", "")
            elif attr.startswith("timeout="):
                node.timeout = attr.replace("timeout=", "")
            elif attr.startswith("proto="):
                node.proto = attr.replace("proto=", "")
                if node.proto == "trpc_trs":
                    node.proto = "trpc-trs"
                elif node.proto == "trpc":
                    node.proto = "trpc-jce2pb"
            elif attr.startswith("server="):
                node.server = attr.replace("server=", "")

        # 生成值
        yield node


# 打开xml文档
file = "./data/VideoRecallClientServer.conf"

# 读取数据
xml_read = read_xml(file)
# 输出头10行记录
# print(xml_read.head(10))
with open("data/recall_client.yaml", "w") as f:
    for line in xml_read:
        f.write(str(line))
        f.write("\n")






