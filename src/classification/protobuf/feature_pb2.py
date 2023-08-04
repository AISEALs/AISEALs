# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: feature.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='feature.proto',
  package='dlserver',
  syntax='proto3',
  serialized_options=_b('\n\026org.tensorflow.exampleB\rFeatureProtosP\001Z;github.com/tensorflow/tensorflow/tensorflow/go/core/example\370\001\001'),
  serialized_pb=_b('\n\rfeature.proto\x12\x08\x64lserver\"\x1a\n\tBytesList\x12\r\n\x05value\x18\x01 \x03(\x0c\"\x1e\n\tFloatList\x12\x11\n\x05value\x18\x01 \x03(\x02\x42\x02\x10\x01\"\x1e\n\tInt64List\x12\x11\n\x05value\x18\x01 \x03(\x03\x42\x02\x10\x01\"\x92\x01\n\x07\x46\x65\x61ture\x12)\n\nbytes_list\x18\x01 \x01(\x0b\x32\x13.dlserver.BytesListH\x00\x12)\n\nfloat_list\x18\x02 \x01(\x0b\x32\x13.dlserver.FloatListH\x00\x12)\n\nint64_list\x18\x03 \x01(\x0b\x32\x13.dlserver.Int64ListH\x00\x42\x06\n\x04kind\"\x7f\n\x08\x46\x65\x61tures\x12\x30\n\x07\x66\x65\x61ture\x18\x01 \x03(\x0b\x32\x1f.dlserver.Features.FeatureEntry\x1a\x41\n\x0c\x46\x65\x61tureEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12 \n\x05value\x18\x02 \x01(\x0b\x32\x11.dlserver.Feature:\x02\x38\x01\"1\n\x0b\x46\x65\x61tureList\x12\"\n\x07\x66\x65\x61ture\x18\x01 \x03(\x0b\x32\x11.dlserver.Feature\"\x98\x01\n\x0c\x46\x65\x61tureLists\x12=\n\x0c\x66\x65\x61ture_list\x18\x01 \x03(\x0b\x32\'.dlserver.FeatureLists.FeatureListEntry\x1aI\n\x10\x46\x65\x61tureListEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.dlserver.FeatureList:\x02\x38\x01\x42i\n\x16org.tensorflow.exampleB\rFeatureProtosP\x01Z;github.com/tensorflow/tensorflow/tensorflow/go/core/example\xf8\x01\x01\x62\x06proto3')
)




_BYTESLIST = _descriptor.Descriptor(
  name='BytesList',
  full_name='dlserver.BytesList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='dlserver.BytesList.value', index=0,
      number=1, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=27,
  serialized_end=53,
)


_FLOATLIST = _descriptor.Descriptor(
  name='FloatList',
  full_name='dlserver.FloatList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='dlserver.FloatList.value', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=55,
  serialized_end=85,
)


_INT64LIST = _descriptor.Descriptor(
  name='Int64List',
  full_name='dlserver.Int64List',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='dlserver.Int64List.value', index=0,
      number=1, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=87,
  serialized_end=117,
)


_FEATURE = _descriptor.Descriptor(
  name='Feature',
  full_name='dlserver.Feature',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='bytes_list', full_name='dlserver.Feature.bytes_list', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='float_list', full_name='dlserver.Feature.float_list', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='int64_list', full_name='dlserver.Feature.int64_list', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='kind', full_name='dlserver.Feature.kind',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=120,
  serialized_end=266,
)


_FEATURES_FEATUREENTRY = _descriptor.Descriptor(
  name='FeatureEntry',
  full_name='dlserver.Features.FeatureEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='dlserver.Features.FeatureEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='dlserver.Features.FeatureEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=330,
  serialized_end=395,
)

_FEATURES = _descriptor.Descriptor(
  name='Features',
  full_name='dlserver.Features',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='feature', full_name='dlserver.Features.feature', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_FEATURES_FEATUREENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=268,
  serialized_end=395,
)


_FEATURELIST = _descriptor.Descriptor(
  name='FeatureList',
  full_name='dlserver.FeatureList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='feature', full_name='dlserver.FeatureList.feature', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=397,
  serialized_end=446,
)


_FEATURELISTS_FEATURELISTENTRY = _descriptor.Descriptor(
  name='FeatureListEntry',
  full_name='dlserver.FeatureLists.FeatureListEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='dlserver.FeatureLists.FeatureListEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='dlserver.FeatureLists.FeatureListEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=528,
  serialized_end=601,
)

_FEATURELISTS = _descriptor.Descriptor(
  name='FeatureLists',
  full_name='dlserver.FeatureLists',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='feature_list', full_name='dlserver.FeatureLists.feature_list', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_FEATURELISTS_FEATURELISTENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=449,
  serialized_end=601,
)

_FEATURE.fields_by_name['bytes_list'].message_type = _BYTESLIST
_FEATURE.fields_by_name['float_list'].message_type = _FLOATLIST
_FEATURE.fields_by_name['int64_list'].message_type = _INT64LIST
_FEATURE.oneofs_by_name['kind'].fields.append(
  _FEATURE.fields_by_name['bytes_list'])
_FEATURE.fields_by_name['bytes_list'].containing_oneof = _FEATURE.oneofs_by_name['kind']
_FEATURE.oneofs_by_name['kind'].fields.append(
  _FEATURE.fields_by_name['float_list'])
_FEATURE.fields_by_name['float_list'].containing_oneof = _FEATURE.oneofs_by_name['kind']
_FEATURE.oneofs_by_name['kind'].fields.append(
  _FEATURE.fields_by_name['int64_list'])
_FEATURE.fields_by_name['int64_list'].containing_oneof = _FEATURE.oneofs_by_name['kind']
_FEATURES_FEATUREENTRY.fields_by_name['value'].message_type = _FEATURE
_FEATURES_FEATUREENTRY.containing_type = _FEATURES
_FEATURES.fields_by_name['feature'].message_type = _FEATURES_FEATUREENTRY
_FEATURELIST.fields_by_name['feature'].message_type = _FEATURE
_FEATURELISTS_FEATURELISTENTRY.fields_by_name['value'].message_type = _FEATURELIST
_FEATURELISTS_FEATURELISTENTRY.containing_type = _FEATURELISTS
_FEATURELISTS.fields_by_name['feature_list'].message_type = _FEATURELISTS_FEATURELISTENTRY
DESCRIPTOR.message_types_by_name['BytesList'] = _BYTESLIST
DESCRIPTOR.message_types_by_name['FloatList'] = _FLOATLIST
DESCRIPTOR.message_types_by_name['Int64List'] = _INT64LIST
DESCRIPTOR.message_types_by_name['Feature'] = _FEATURE
DESCRIPTOR.message_types_by_name['Features'] = _FEATURES
DESCRIPTOR.message_types_by_name['FeatureList'] = _FEATURELIST
DESCRIPTOR.message_types_by_name['FeatureLists'] = _FEATURELISTS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

BytesList = _reflection.GeneratedProtocolMessageType('BytesList', (_message.Message,), dict(
  DESCRIPTOR = _BYTESLIST,
  __module__ = 'feature_pb2'
  # @@protoc_insertion_point(class_scope:dlserver.BytesList)
  ))
_sym_db.RegisterMessage(BytesList)

FloatList = _reflection.GeneratedProtocolMessageType('FloatList', (_message.Message,), dict(
  DESCRIPTOR = _FLOATLIST,
  __module__ = 'feature_pb2'
  # @@protoc_insertion_point(class_scope:dlserver.FloatList)
  ))
_sym_db.RegisterMessage(FloatList)

Int64List = _reflection.GeneratedProtocolMessageType('Int64List', (_message.Message,), dict(
  DESCRIPTOR = _INT64LIST,
  __module__ = 'feature_pb2'
  # @@protoc_insertion_point(class_scope:dlserver.Int64List)
  ))
_sym_db.RegisterMessage(Int64List)

Feature = _reflection.GeneratedProtocolMessageType('Feature', (_message.Message,), dict(
  DESCRIPTOR = _FEATURE,
  __module__ = 'feature_pb2'
  # @@protoc_insertion_point(class_scope:dlserver.Feature)
  ))
_sym_db.RegisterMessage(Feature)

Features = _reflection.GeneratedProtocolMessageType('Features', (_message.Message,), dict(

  FeatureEntry = _reflection.GeneratedProtocolMessageType('FeatureEntry', (_message.Message,), dict(
    DESCRIPTOR = _FEATURES_FEATUREENTRY,
    __module__ = 'feature_pb2'
    # @@protoc_insertion_point(class_scope:dlserver.Features.FeatureEntry)
    ))
  ,
  DESCRIPTOR = _FEATURES,
  __module__ = 'feature_pb2'
  # @@protoc_insertion_point(class_scope:dlserver.Features)
  ))
_sym_db.RegisterMessage(Features)
_sym_db.RegisterMessage(Features.FeatureEntry)

FeatureList = _reflection.GeneratedProtocolMessageType('FeatureList', (_message.Message,), dict(
  DESCRIPTOR = _FEATURELIST,
  __module__ = 'feature_pb2'
  # @@protoc_insertion_point(class_scope:dlserver.FeatureList)
  ))
_sym_db.RegisterMessage(FeatureList)

FeatureLists = _reflection.GeneratedProtocolMessageType('FeatureLists', (_message.Message,), dict(

  FeatureListEntry = _reflection.GeneratedProtocolMessageType('FeatureListEntry', (_message.Message,), dict(
    DESCRIPTOR = _FEATURELISTS_FEATURELISTENTRY,
    __module__ = 'feature_pb2'
    # @@protoc_insertion_point(class_scope:dlserver.FeatureLists.FeatureListEntry)
    ))
  ,
  DESCRIPTOR = _FEATURELISTS,
  __module__ = 'feature_pb2'
  # @@protoc_insertion_point(class_scope:dlserver.FeatureLists)
  ))
_sym_db.RegisterMessage(FeatureLists)
_sym_db.RegisterMessage(FeatureLists.FeatureListEntry)


DESCRIPTOR._options = None
_FLOATLIST.fields_by_name['value']._options = None
_INT64LIST.fields_by_name['value']._options = None
_FEATURES_FEATUREENTRY._options = None
_FEATURELISTS_FEATURELISTENTRY._options = None
# @@protoc_insertion_point(module_scope)
