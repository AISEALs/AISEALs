# import numpy as np
import struct
import logging
import crcmod
# import feature_pb2
# import example_pb2
try:
    from crc32c import crc32
except ImportError:
    crc32 = None


# def _encoded_int64_feature(ndarray):
#     return feature_pb2.Feature(int64_list=feature_pb2.Int64List(
#         value=ndarray.flatten().tolist()))
#
#
# def _encoded_bytes_feature(tf_encoded):
#     encoded = tf_encoded.eval()
#
#     def string_to_bytes(value):
#         return feature_pb2.BytesList(value=[value])
#
#     return feature_pb2.Feature(bytes_list=string_to_bytes(encoded))
#
#
# def _string_feature(value):
#     value = value.encode('utf-8')
#     return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=[value]))

def _default_crc32c_fn(value):
    """Calculates crc32c by either snappy or crcmod based on installation."""

    if not _default_crc32c_fn.fn:
        try:
            import snappy  # pylint: disable=import-error
            _default_crc32c_fn.fn = snappy._crc32c  # pylint: disable=protected-access
        except ImportError:
            logging.warning('Couldn\'t find python-snappy so the implementation of '
                            '_TFRecordUtil._masked_crc32c is not as fast as it could '
                            'be.')
            _default_crc32c_fn.fn = crcmod.predefined.mkPredefinedCrcFun('crc-32c')
    return _default_crc32c_fn.fn(value)

_default_crc32c_fn.fn = None

def _masked_crc32c(value, crc32c_fn=_default_crc32c_fn):
    """Compute a masked crc32c checksum for a value.

    Args:
      value: A string for which we compute the crc.
      crc32c_fn: A function that can compute a crc32c.
        This is a performance hook that also helps with testing. Callers are
        not expected to make use of it directly.
    Returns:
      Masked crc32c checksum.
    """
    if isinstance(value, str):
        value = value.encode()
    crc = crc32c_fn(value)
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff

def write_record(file_handle, value):
    """Encode a value as a TFRecord.

    Args:
      file_handle: The file to write to.
      value: A string content of the record.
    """
    encoded_length = struct.pack('<Q', len(value))
    tf_data = bytearray(b'').join([
        encoded_length,
        struct.pack('<I', _masked_crc32c(encoded_length)),
        value,
        struct.pack('<I', _masked_crc32c(value))])

    file_handle.write(tf_data)