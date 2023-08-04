import numpy as np
import pandas as pd


tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'one', 'one', 'two',
                     'one', 'two', 'one', 'two']]))

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
#why this error?
# print(df['bar', 'one'])

stacked = df.stack()

s = pd.Series(np.random.randn(8), index=index)
print(s['bar', 'one'])

keys = np.array([
    ['A', 'B'],
    ['A', 'B'],
    ['A', 'B'],
    ['A', 'B'],
    ['C', 'D'],
    ['C', 'D'],
    ['C', 'D'],
    ['E', 'F'],
    ['E', 'F'],
    ['G', 'H']
    ])

df = pd.DataFrame(
        np.hstack([keys, np.random.randn(10, 4).round(2)]),
             columns = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6']
    )

df[['col3', 'col4', 'col5', 'col6']] = \
        df[['col3', 'col4', 'col5', 'col6']].astype(float)
