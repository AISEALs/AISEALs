#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections
import timeit
from json import dumps as json_dumps
from typing import List

import numpy as np

from .ascii_table import ascii_table


class States:
    sum: float
    count: int
    max: float
    states: List

    def __init__(
        self,
        sum: float = 0.0,
        count: int = 0,
        max: float = -float("inf"),
        states: List = None,
    ):
        self.sum = sum
        self.count = count
        self.max = max
        self.states = [] if states is None else states

    @property
    def average(self):
        return self.sum / (self.count or 1)

    @property
    def p50(self):
        return np.percentile(self.states, 50)

    @property
    def p90(self):
        return np.percentile(self.states, 90)

    @property
    def p99(self):
        return np.percentile(self.states, 99)

    def add(self, state):
        self.states.append(state)
        self.sum += state
        self.count += 1
        self.max = max(self.max, state)


SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = 60 * SECONDS_IN_MINUTE
SECONDS_IN_DAY = 24 * SECONDS_IN_HOUR


def format_time(seconds):
    if seconds > 60:
        days, seconds = int(seconds // SECONDS_IN_DAY), seconds % SECONDS_IN_DAY
        hours, seconds = int(seconds // SECONDS_IN_HOUR), seconds % SECONDS_IN_HOUR
        minutes, seconds = (
            int(seconds // SECONDS_IN_MINUTE),
            seconds % SECONDS_IN_MINUTE,
        )
        if days:
            if minutes >= 30:
                hours += 1
            return f"{days}d{hours}h"
        elif hours:
            if seconds >= 30:
                minutes += 1
            return f"{hours}h{minutes}m"
        else:
            seconds = int(round(seconds))
            return f"{minutes}m{seconds}s"
    elif seconds > 1:
        return f"{seconds:.1f}s"
    elif seconds > 0.001:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds * 1000000:.1f}ns"


class Snapshot:
    def __init__(self, states):
        self.states = states
        self.start = timeit.default_timer()

    def report(self, report_pep=False):
        snapshot_total = timeit.default_timer() - self.start

        def path(key):
            return " -> ".join(label for label, _ in key)

        def print_pep(results, snapshot_total):
            for key, times in sorted(self.states.items()):
                if path(key) == "evaluate -> pytorch eval once":
                    info = {
                        "type": path(key),
                        "metric": "latency",
                        "unit": "ms",
                        "value": f"{times.average * 1000:.1f}",
                    }
                    print("PyTorchObserver " + json_dumps(info))

        if len(self.states) == 0:
            print(
                "Note: Nothing was reported. "
                'Please use timing.time("foo") to measure time.'
            )
            return

        results = [
            {
                "name": key,
                "total": datas.sum,
                "avg": "%.2f" % datas.average,
                "max": datas.max,
                "p50": datas.p50,
                "p90": datas.p90,
                "p99": datas.p99,
                "count": datas.count,
            }
            for key, datas in sorted(self.states.items())
        ]
        print(
            ascii_table(
                results,
                human_column_names={
                    "name": "Stage",
                    "total": "Total",
                    "avg": "Average",
                    "max": "Max",
                    "p50": "P50",
                    "p90": "P90",
                    "p99": "P99",
                    "count": "Count",
                },
                # footer={"name": "Total time", "total": format_time(snapshot_total)},
                alignments={"name": "<"},
            )
        )
        if report_pep:
            print_pep(results, snapshot_total)


class HierarchicalStatus(States):
    def __init__(self, label):
        super(HierarchicalStatus, self).__init__()
        self.label = None

    def push(self, label, data):
        self.label = label



status_dict = dict()
snapshot = Snapshot(status_dict)
report = snapshot.report

def record(label, l):
    states = States()
    for i in l:
        states.add(i)

    status_dict[label] = states



