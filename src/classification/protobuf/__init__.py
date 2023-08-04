# try:
#     import tensorflow as tf
#     pb2_Features = tf.train.Features
#     pb2_Feature = tf.train.Feature
#     pb2_Int64List = tf.train.Int64List
#     pb2_Example = tf.train.Example
# except:
from protobuf import feature_pb2, example_pb2
pb2_Features = feature_pb2.Features
pb2_Feature = feature_pb2.Feature
pb2_Int64List = feature_pb2.Int64List
pb2_Example = example_pb2.Example
pb2_BytesList = feature_pb2.BytesList

__all__ = [pb2_Feature, pb2_Features, pb2_Int64List, pb2_Example, pb2_BytesList]
