
file_name = '/Users/jiananliu/itime_p30_p70.txt'
result = [[] for _ in range(3)]
with open(file_name) as f:
    for line in f:
        sp = line.split('\t')
        for i, col in enumerate(sp):
            result[i].append(col.strip())


for x in result:
    print(','.join(map(str, x)))
    print(len(x))


