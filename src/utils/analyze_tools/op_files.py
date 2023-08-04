import glob
import pandas as pd


# read multi file by regex string
# eg:file_names = r'./data/各层粉丝量级的作者量分布*.csv'
def read_multi_file(file_names):
    all_files = glob.glob(file_names)
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        # df.drop(df.tail(2).index, inplace=True)
        li.append(df)
    df = pd.concat(li, axis=0, ignore_index=True)
    return df


if __name__ == '__main__':
    file_names = r'./data/各层粉丝量级的作者量分布*.csv'
    print(read_multi_file(file_names))
