import heapq2

h = []
heapq2.heappush(h, (3, 'i'))
heapq2.heappush(h, (1, 'w'))
heapq2.heappush(h, (2, 'i'))

while len(h) > 0:
    print(heapq2.heappop(h))
# print(h[0])
