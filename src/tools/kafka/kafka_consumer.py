#encode=utf8
from kafka import KafkaConsumer
from conf import config

# 这个是producter发送的kafka信息，可以用于测试是否发送成功。
# TOPIC = "hdp_ubu_spider_tribe_labels"
# CLIENT_ID = "hdp_ubu_spider-hdp_ubu_spider_tribe_labels-BHDBs"
# GROUP_ID = "hdp_ubu_spider-hdp_ubu_spider_tribe_labels-BHDBs-1"

logger = config.logger

class kafka_consumer:
    def __init__(self,
                 topic,
                 client_id,
                 group_id,
                 bootstrap_servers,
                 enable_auto_commit=True,
                 max_poll_interval_ms=100000):
        if isinstance(bootstrap_servers, str):
            bootstrap_servers = bootstrap_servers.split(",")
        self.consumer = KafkaConsumer(topic,
                                      client_id=client_id,
                                      group_id=group_id,
                                      auto_offset_reset='earliest',
                                      bootstrap_servers=bootstrap_servers,
                                      enable_auto_commit=enable_auto_commit,
                                      max_poll_interval_ms=max_poll_interval_ms)  #参数为接收主题和kafka服务器地址
        logger.info("partitions: {}".format(self.consumer.partitions_for_topic(topic=topic)))  #获取test主题的分区信息
        # logger.info("topics: {}".format(self.consumer.topics()))  #获取主题列表
        logger.info("subscription: {}".format(self.consumer.subscription()))  #获取当前消费者订阅的主题
        # print(consumer.assignment())  #获取当前消费者topic、分区信息

    def get_messages(self):
        # # 这是一个永久堵塞的过程，生产者消息会缓存在消息队列中,并且不删除,所以每个消息在消息队列中都有偏移
        for message in self.consumer:  # consumer是一个消息队列，当后台有消息时，这个消息队列就会自动增加．所以遍历也总是会有数据，当消息队列中没有数据时，就会堵塞等待消息带来
            logger.info("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,message.offset, message.key,message.value))
            # todo: consumer message
            yield message

    def get_messages_by_poll(self, timeout_ms=1000, max_records=100):
        while True:
            yield self.consumer.poll(timeout_ms=timeout_ms, max_records=max_records)

    def simulate_poll(self):
        data = """{"infoContent":"东莞的朋友们，大家帮忙留意一下这个人，她叫鲁金兰，云南省楚雄华南县人，5月24下午从清远银盏出来至今未归，听说在东莞的某个地方和一个云南省的男的一起跑的，希望见过的告知","infoTitle":"","infoid":1001897216129560576,"insertTime":1559552510639,"local":700000001,"localName":"万能求助圈","messageType":1,"pics":["https://pic7.58cdn.com.cn/mobile/big/n_v29e36f6dd44f3453492689e5074cf7592.jpg","https://pic7.58cdn.com.cn/mobile/big/n_v2a8387452b19c4c14b93096ee0d71738a.jpg","https://pic7.58cdn.com.cn/mobile/big/n_v205fef51ee1b940b3b7c64391a7788a4b.jpg"],"tagsByMachine":[],"topicId":0,"uid":29724360578312}"""
        yield {"hdp_ubu_wuxian_machinelabel": [data, data]}

    def close(self):
        self.consumer.close()

    def commit(self):
        # self.consumer.commit_async()
        self.consumer.commit()

    def seek_by_lastest_offset(self, topicPartition):
        lastOffSet = self.consumer.committed(topicPartition)
        self.consumer.seek(topicPartition, lastOffSet)


if __name__ == '__main__':
    BOOTSTRAP_ERVERS = "10.135.9.4:9092,10.135.9.5:9092,10.135.9.6:9092,10.135.9.7:9092,10.135.9.8:9092"
    TOPIC = "hdp_ubu_wuxian_machinelabel"
    CLIENT_ID = "hdp_ubu_wuxian-hdp_ubu_wuxian_machinelabel-OXjXt"
    GROUP_ID = "hdp_ubu_wuxian-hdp_ubu_wuxian_machinelabel-OXjXt-1"

    import time
    t = time.time()
    ts = int(round(t * 1000))
    consumer = kafka_consumer(topic=TOPIC, client_id=CLIENT_ID, group_id=GROUP_ID, bootstrap_servers=BOOTSTRAP_ERVERS, max_poll_interval_ms=4000, enable_auto_commit=False)

    i = 0
    failed_times = 0
    for topic2messages in consumer.get_messages_by_poll():
        for topic2partition in topic2messages.keys():
            consumer.seek_by_lastest_offset(topic2partition)
        i += 1
        messages = [mess for mess_list in topic2messages.values() for mess in mess_list]
        if len(messages) > 0:
            print("{} {} {} {}".format(messages[0].offset, messages[0].partition, messages[0].topic, messages[0].timestamp))
            # if i > 10 and failed_times < 3:
            if failed_times < 3:
                # i = 0
                failed_times += 1
                import time
                print("sleep 5s:")
                # time.sleep(5)
                print("commit failed")
                # exit(0)
                from kafka.structs import TopicPartition
                tp = TopicPartition(messages[0].topic, messages[0].partition)
                consumer.seek_by_lastest_offset(tp)
                continue
        try:
            consumer.commit()
            failed_times = 0
            i = 0
            print("commit success")
        except Exception as ex:
            logger.error("error: {}".format(str(ex)))
            # continue

    # consumer.close()

