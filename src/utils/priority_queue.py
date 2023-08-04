import heapq
from collections import Counter
import time


class PriorityQueue(object):
  """实现一个优先级队列，每次pop优先级最高的元素"""
  def __init__(self, maxsize, asce=True):
    self.pq = []                      # list of entries arranged in a heap
    self.entry_finder = {}            # mapping of tasks to entries
    self.embedding_idx = {}           # task -> embedding_id
    self.REMOVED = '<removed-task>'   # placeholder for a removed task
    self.counter = Counter()
    self.asce = asce                  # 默认升序
    self.maxsize = maxsize
    assert (maxsize > 0)

  def init_from_dict(self, embedding_idx):
    for task, idx in embedding_idx.items():
      if idx < self.maxsize:
        self.push(task, idx=idx)

  def look_up(self, task):
    if not isinstance(task, str):
      task = str(task)
    record = {}
    self.push(task, record=record)
    old_task = record.get(task, -1)
    replace_embedding_idx = self.embedding_idx[task] if old_task != -1 else -1
    rs = (self.embedding_idx[task], self.entry_finder[task][1], replace_embedding_idx)
    return rs

  # embedding_idx 改变情形
  # 1.对列没有满，新增一个, add {task, len(entry_finder)}
  # 2.对列满了，新增一个, update(old_task, task)
  def push(self, task, priority=0, idx=None, record=None):
    if not isinstance(task, str):
      task = str(task)
    if priority == 0:
      priority = int(time.time())
    if self.full() and task not in self.entry_finder:
      poped_task = self.pop()
      self._push(task, priority, idx)
      embedding_idx = self.embedding_idx.pop(poped_task)
      self.embedding_idx[task] = embedding_idx
      if record != None and isinstance(record, dict):
        record[task] = poped_task
    else:
      self._push(task, priority, idx)

  def _push(self, task, priority, idx=None):
    'Add a new task or update the priority of an existing task'
    if task in self.entry_finder:
      self.remove(task)
    else:
      self.embedding_idx[task] = idx if idx is not None else len(self.entry_finder)
    # count = next(self.counter)
    self.counter[task] += 1
    entry = [priority, self.counter[task], task]
    self.entry_finder[task] = entry
    heapq.heappush(self.pq, entry)

  def remove(self, task):
    'Mark an existing task as REMOVED.  Raise KeyError if not found.'
    entry = self.entry_finder.pop(task)
    entry[-1] = self.REMOVED

  def pop(self):
    'Remove and return the lowest priority task. Raise KeyError if empty.'
    while self.pq:
      priority, count, task = heapq.heappop(self.pq)
      if task is not self.REMOVED:
        del self.entry_finder[task]
        return task
    raise KeyError('pop from an empty priority queue')

  def full(self):
    if self.maxsize <= 0:
        raise Exception('queue must be maxisze > 0')
    return len(self.entry_finder) >= self.maxsize

  def __len__(self):
    if self.maxsize <= 0:
      raise Exception('queue must be maxisze > 0')
    return self.maxsize

if __name__ == '__main__':
  pqueue = PriorityQueue(maxsize=4)
  pqueue.push('d')
  pqueue.push('f')
  pqueue.push('a')
  pqueue.push(1)
  print(pqueue.look_up("a"))
  print(pqueue.look_up("b"))
  # print(pqueue.pop())
  # print(pqueue.pop())
  # print(pqueue.pop())
