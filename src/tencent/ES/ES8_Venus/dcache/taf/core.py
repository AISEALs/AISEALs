# -*- encoding=utf-8 -*-

from taf.__util import util
from taf.__jce import JceInputStream
from taf.__jce import JceOutputStream
from taf.__wup import TafUniPacket


class tafcore:

    class JceInputStream(JceInputStream):
        pass

    class JceOutputStream(JceOutputStream):
        pass

    class TafUniPacket(TafUniPacket):
        pass

    class boolean(util.boolean):
        pass

    class int8(util.int8):
        pass

    class uint8(util.uint8):
        pass

    class int16(util.int16):
        pass

    class uint16(util.uint16):
        pass

    class int32(util.int32):
        pass

    class uint32(util.uint32):
        pass

    class int64(util.int64):
        pass

    class float(util.float):
        pass

    class double(util.double):
        pass

    class bytes(util.bytes):
        pass

    class string(util.string):
        pass

    class struct(util.struct):
        pass

    @staticmethod
    def mapclass(ktype, vtype): return util.mapclass(ktype, vtype)

    @staticmethod
    def vctclass(vtype): return util.vectorclass(vtype)

    @staticmethod
    def printHex(buff): util.printHex(buff)
