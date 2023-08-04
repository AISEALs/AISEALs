#coding=utf-8
import sys
sys.path.append("../")
import json
import traceback
from tools.kafka.kafka_consumer import kafka_consumer
from tools.kafka.kafka_producer import kafka_producer
from common_tools.logger import get_logger
from tools.timer import timer

logger = get_logger("log/finnal_predict_labels.log")
logger.info("predict labels starting...................")

from text_label import *
logger.info("loading model finish....................")

def run():
    BOOTSTRAP_ERVERS = "10.135.9.4:9092,10.135.9.5:9092,10.135.9.6:9092,10.135.9.7:9092,10.135.9.8:9092"
    CONSUMER_TOPIC = "hdp_ubu_wuxian_machinelabel"
    CONSUMER_CLIENT_ID = "hdp_ubu_wuxian-hdp_ubu_wuxian_machinelabel-OXjXt"
    CONSUMER_GROUP_ID = "hdp_ubu_wuxian-hdp_ubu_wuxian_machinelabel-OXjXt-1" if config.debug_mode else "hdp_ubu_wuxian-hdp_ubu_wuxian_machinelabel-OXjXt-2"

    consumer = kafka_consumer(CONSUMER_TOPIC, CONSUMER_CLIENT_ID, CONSUMER_GROUP_ID, BOOTSTRAP_ERVERS, enable_auto_commit=False)

    PRODUCER_CLIENT_ID = "hdp_ubu_spider-hdp_ubu_spider_tribe_labels-BHDBs"
    PRODUCER_TOPIC = "hdp_ubu_spider_tribe_labels"

    producer = kafka_producer(BOOTSTRAP_ERVERS, PRODUCER_CLIENT_ID)
    failed_times = 0
    try:
        # for topic2messages in consumer.simulate_poll(): #for test
        for topic2messages in consumer.get_messages_by_poll(max_records=10):
            try:
                with timer("Timer predict labels", logger):
                    messages = [mess for mess_list in topic2messages.values() for mess in mess_list]
                    logger.info("read from topic:[{}] len:{}".format(CONSUMER_TOPIC, len(messages)))
                    for (tp, mess) in topic2messages.items():
                        logger.info("{} {} {} {}".format(mess[0].offset, mess[0].partition, mess[0].topic, mess[0].timestamp))

                    if len(messages) == 0:
                        continue
                    contents = list(map(lambda x: x.value, messages))
                    mess_dicts, infoids, infoContents, infoTitles, localNames, pics = parse_message(mess_list=contents)
                    logger.info("parse message topic:[{}] len:{} => len:{} ".format(CONSUMER_TOPIC, len(messages), len(infoids)))

                    results = predict_label(list(zip(infoids, pics, infoContents)))

                    if len(infoids) != len(results):
                        logger.error("predict_label result len:{} but should return len:{} failed_times:{}".format(len(results), len(infoids), failed_times))
                        if failed_times < 2:
                            failed_times += 1
                            for tp in topic2messages.keys():
                                consumer.seek_by_lastest_offset(tp)
                            time.sleep(10)
                            continue
                    logger.info("predict_label get result num:{}".format(len(results)))

                consumer.commit()# 每次poll处理完显示commit一下（poll只有在close时才会commit，程序在kill的情况下，finnal模块不会走，出现不提交offset）
                failed_times = 0
            except Exception as ex:
                logger.error("error: {}".format(str(ex)))
                continue

            for (infoid, result, mess_dict) in zip(infoids, results, mess_dicts):
                labels = result[0] if isinstance(result[0], list) else [result[0]]
                mess_dict['tagsByMachine'] = list(map(int, labels))
                logger.info("send kafka data:{}".format(mess_dict))
                value = json.dumps(mess_dict)
                if not config.debug_mode:
                    producer.send(PRODUCER_TOPIC, key=str(infoid).encode('utf8'), value=value.encode('utf8'))
                else:
                    print("info_id:{} value:{}".format(infoid, value))

            producer.flush()

    except Exception as ex:
        logger.error("error: {}".format(str(ex)))
        logger.error(str(traceback.print_exc(ex)))
    finally:
        consumer.close()
        producer.close()
        logger.info("predict labels finish...................")


def parse_message(mess_list):
    mess_dicts = list(map(json.loads, mess_list))
    infoids = []
    infoContents = []
    infoTitles = []
    localNames = []
    pics = []
    for mess_dict in mess_dicts:
        try:
            logger.info("recv from kafka data:{}".format(mess_dict))
            if 'infoid' not in mess_dict or 'infoContent' not in mess_dict or 'pics' not in mess_dict:
                logger.warn("json: {}\n has no infoid, infoContent or pics!".format(str(mess_dict)))
            infoids.append(mess_dict['infoid'])
            from tools.tools import clean_html
            infoConent = mess_dict['infoContent'].replace("\n", "")
            infoContents.append(" ".join([clean_html(i) for i in jieba.cut(infoConent)]))
            pics.append(mess_dict['pics'])

            infoTitle = mess_dict if 'infoTitle' in mess_dict else ""
            infoTitles.append(infoTitle)
            localName = mess_dict['localName'] if 'localName' in mess_dict else ""
            localNames.append(localName)
        except Exception as ex:
            logger.error("error: {}".format(str(ex)))
            continue

    if not (len(mess_list) == len(infoids) == len(infoContents) == len(infoTitles) == len(localNames) == len(pics)):
        logger.error("parse message failed: {} {} {} {} {} {}".format(len(mess_dicts), len(infoids), len(infoContents), len(infoTitles), len(localNames), len(pics)))

    return mess_dicts, infoids, infoContents, infoTitles, localNames, pics


def test_json():
    data = """{"infoContent":"东莞的朋友们，大家帮忙留意一下这个人，她叫鲁金兰，云南省楚雄华南县人，5月24下午从清远银盏出来至今未归，听说在东莞的某个地方和一个云南省的男的一起跑的，希望见过的告知","infoTitle":"","infoid":1001897216129560576,"insertTime":1559552510639,"local":700000001,"localName":"万能求助圈","messageType":1,"pics":["https://pic7.58cdn.com.cn/mobile/big/n_v29e36f6dd44f3453492689e5074cf7592.jpg","https://pic7.58cdn.com.cn/mobile/big/n_v2a8387452b19c4c14b93096ee0d71738a.jpg","https://pic7.58cdn.com.cn/mobile/big/n_v205fef51ee1b940b3b7c64391a7788a4b.jpg"],"tagsByMachine":[],"topicId":0,"uid":29724360578312}"""

    data_json = json.loads(data)

    print(data_json)

if __name__ == '__main__':
    # test_json()
    run()





