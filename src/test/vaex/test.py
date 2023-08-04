import vaex
import numpy as np

@vaex.register_function()
def add_one(ar):
    return ar + 1

df = vaex.from_arrays(x=np.arange(4))
df.func.add_one(df.x)

import numpy as np
df = vaex.from_arrays(departure=np.arange('2015-01-02', '2015-12-06', dtype='datetime64'), col2=np.arange('2015-01-01', '2015-12-05', dtype='datetime64'))


@vaex.register_function(as_property=True, scope='dt')
def dt_relative_day(x):
    return vaex.functions.dt_dayofyear(x)/365.

@vaex.register_function(as_property=True, scope='dt')
def func1(x):
    return vaex.functions.td_days(x)

df.func.dt_relative_day(df.departure)

df['col3'] = (df.departure - df.col2).dt.func1
# df['col3'] = df.col3.dt.func1