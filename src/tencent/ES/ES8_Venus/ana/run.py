import sys

for line in sys.stdin:
    seps = line.strip().split("_")
    if len(seps) != 5:
        continue
    pt, vv, fn= float(seps[0]), float(seps[3]), int(seps[4])
    pt /= fn
    vv /= fn
    print(pow(pt,0.8), pow(vv,2), fn)
    
