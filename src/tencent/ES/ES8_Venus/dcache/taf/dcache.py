from taf.core import tafcore
from taf.__rpc import ServantProxy


# proxy for client
class dcacheProxy(ServantProxy):

    def getString(self, moduleName, keyItem, context=ServantProxy.mapcls_context()):
        oos = tafcore.JceOutputStream()
        oos.write(tafcore.string, 1, moduleName)
        oos.write(tafcore.string, 2, keyItem)
        rsp = self.taf_invoke(ServantProxy.JCENORMAL, "getString",
                              oos.getBuffer(), context, None)
        ios = tafcore.JceInputStream(rsp.sBuffer)
        ret = ios.read(tafcore.int32, 0, True)
        value = ios.read(tafcore.string, 3, True)
        return (ret, value)

    def setString(self, moduleName, keyItem, value, context=ServantProxy.mapcls_context()):
        oos = tafcore.JceOutputStream()
        oos.write(tafcore.string, 1, moduleName)
        oos.write(tafcore.string, 2, keyItem)
        oos.write(tafcore.string, 3, value)

        rsp = self.taf_invoke(ServantProxy.JCENORMAL, "setString",
                              oos.getBuffer(), context, None)

        ios = tafcore.JceInputStream(rsp.sBuffer)
        ret = ios.read(tafcore.int32, 0, True)

    def setStringBatch(self, moduleName, keyValue, context=ServantProxy.mapcls_context()):
        oos = tafcore.JceOutputStream()
        oos.write(tafcore.string, 1, moduleName)
        oos.write(tafcore.mapclass(tafcore.string, tafcore.string), 2, keyValue)

        rsp = self.taf_invoke(ServantProxy.JCENORMAL, "setStringBatch",
                              oos.getBuffer(), context, None)

        ios = tafcore.JceInputStream(rsp.sBuffer)
        ret = ios.read(tafcore.int32, 0, True)
        keyResult = ios.read(tafcore.mapclass(tafcore.string, tafcore.int32), 3, True)

        return (ret, keyResult)

    def setStringEx(self, moduleName, keyItem, value, ver, dirty, expireTimeSecond, context = ServantProxy.mapcls_context()):
        oos = tafcore.JceOutputStream();
        oos.write(tafcore.string, 1, moduleName);
        oos.write(tafcore.string, 2, keyItem);
        oos.write(tafcore.string, 3, value);
        oos.write(tafcore.int8, 4, ver);
        oos.write(tafcore.boolean, 5, dirty);
        oos.write(tafcore.int32, 6, expireTimeSecond);

        rsp = self.taf_invoke(ServantProxy.JCENORMAL, "setStringEx", oos.getBuffer(), context, None);

        ios = tafcore.JceInputStream(rsp.sBuffer);
        ret = ios.read(tafcore.int32, 0, True);

        return (ret);

    def delString(self, moduleName, keyItem, context = ServantProxy.mapcls_context()):
        oos = tafcore.JceOutputStream();
        oos.write(tafcore.string, 1, moduleName);
        oos.write(tafcore.string, 2, keyItem);

        rsp = self.taf_invoke(ServantProxy.JCENORMAL, "delString", oos.getBuffer(), context, None);

        ios = tafcore.JceInputStream(rsp.sBuffer);
        ret = ios.read(tafcore.int32, 0, True);

        return (ret);

    def delInt(self, moduleName, keyItem, context = ServantProxy.mapcls_context()):
        oos = tafcore.JceOutputStream();
        oos.write(tafcore.string, 1, moduleName);
        oos.write(tafcore.int32, 2, keyItem);

        rsp = self.taf_invoke(ServantProxy.JCENORMAL, "delInt", oos.getBuffer(), context, None);

        ios = tafcore.JceInputStream(rsp.sBuffer);
        ret = ios.read(tafcore.int32, 0, True);

        return (ret);

    def delLong(self, moduleName, keyItem, context = ServantProxy.mapcls_context()):
        oos = tafcore.JceOutputStream();
        oos.write(tafcore.string, 1, moduleName);
        oos.write(tafcore.int64, 2, keyItem);

        rsp = self.taf_invoke(ServantProxy.JCENORMAL, "delLong", oos.getBuffer(), context, None);

        ios = tafcore.JceInputStream(rsp.sBuffer);
        ret = ios.read(tafcore.int32, 0, True);

        return (ret);
