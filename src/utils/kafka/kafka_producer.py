from kafka import KafkaProducer
import logging

# logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
from conf import config

logger = config.logger

class kafka_producer(object):
    def __init__(self, bootstrap_servers, client_id):
        if isinstance(bootstrap_servers, str):
            bootstrap_servers = bootstrap_servers.split(",")
        self.producer = KafkaProducer(client_id=client_id, bootstrap_servers=bootstrap_servers)

    # # Asynchronous by default
    # # key 目的是把固定doc_id的帖子发送到同一个kafka partition中，防止乱序
    # future = producer.send(TOPIC, value=doc_data, key=doc_id)
    #
    # # Block for 'synchronous' sends
    # try:
    #     record_metadata = future.get(timeout=10)
    # except KafkaError:
    #     # Decide what to do if produce request failed...
    #     print("process data to kafka failed!")
    #     pass
    #
    # # Successful result returns assigned partition and offset
    # print (record_metadata.topic)
    # print (record_metadata.partition)
    # print (record_metadata.offset)

    # # produce keyed messages to enable hashed partitioning
    # producer.send(TOPIC, key=b'foo', value=b'bar')
    #
    # # encode objects via msgpack
    # producer = KafkaProducer(value_serializer=msgpack.dumps)
    # producer.send('msgpack-topic', {'key': 'value'})
    #
    # # produce json messages
    # producer = KafkaProducer(value_serializer=lambda m: json.dumps(m).encode('ascii'))
    # producer.send('json-topic', {'key': 'value'})
    #
    # produce asynchronously
    # for _ in range(100):
    #     producer.send('my-topic', b'msg')

    def send(self, topic, key, value):
        def on_send_success(record_metadata):
            # print(record_metadata.topic)
            # print(record_metadata.partition)
            logger.info("offset: {} success".format(record_metadata.offset))

        def on_send_error(excp):
            logger.error('I am an errback', exc_info=excp)
            # handle exception
            pass

        # produce asynchronously with callbacks
        self.producer.send(topic, value=value, key=key).add_callback(on_send_success).add_errback(on_send_error)

    def flush(self):
        # block until all async messages are sent
        self.producer.flush()

        # configure multiple retries
        # producer = KafkaProducer(retries=5)

    def close(self):
        self.producer.close()


if __name__ == '__main__':
    BOOTSTRAP_ERVERS = "10.135.9.4:9092,10.135.9.5:9092,10.135.9.6:9092,10.135.9.7:9092,10.135.9.8:9092"
    CLIENT_ID = "hdp_ubu_spider-hdp_ubu_spider_tribe_labels-BHDBs"
    TOPIC = "hdp_ubu_spider_tribe_labels"

    producer = kafka_producer(BOOTSTRAP_ERVERS, CLIENT_ID)
    producer.send(TOPIC, key=1, value="my_data")
    producer.flush()
