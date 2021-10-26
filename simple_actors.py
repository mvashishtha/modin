from re import split
from modin.config import NPartitions
from modin.distributed.dataframe.pandas import partitions, unwrap_partitions
from modin.distributed.dataframe.pandas import from_partitions
import numpy as np
import pandas
import ray
import modin.pandas as pd
import time
import asyncio

ray.init(num_cpus=8)
num_partitions = 8
num_actors = 8
NPartitions.put(num_partitions)


@ray.remote
class ShuffleActor(object):
    def __init__(self, pos):
        self.other_actors = None
        self.num_position = pos
        self.list_of_dfs = []

    def add_pool(self, other_actors):
        self.other_actors = other_actors
        self.splits = list(range(len(other_actors)))

    def append_df(self, df):
        self.list_of_dfs.append(df)
        # print(f"finished appending value {df} to actor {self.num_position}")

    def append_splits(self):
        # self.append_df(self.num_position)
        # return [
        #     actor.append_df.remote(self.num_position)
        #     for i, actor in enumerate(self.other_actors)
        #     if i != self.num_position
        # ]
        self.append_df(self.num_position)
        return [
            actor.append_df.remote(self.num_position)
            for i, actor in enumerate(self.other_actors)
            if i != self.num_position
        ]

    def split_df(self):
        print(
            f"starting split_df for actor {self.num_position} at {time.time() - start}"
        )
        return 1

    def print_list_of_dfs(self):
        print(
            f"splits for actor {self.num_position}: {', '.join(str(x) for x in self.list_of_dfs)}"
        )


start = time.time()
shuffle_actors = [ShuffleActor.remote(i) for i in range(num_actors)]
ray.get([actor.add_pool.remote(shuffle_actors) for actor in shuffle_actors])
print(f"time to add pools: {time.time() - start}")
ray.get([actor.split_df.remote() for actor in shuffle_actors])
print(f"Time to split dfs: {time.time() - start}")
ray.get(
    [
        ref
        for sublist in ray.get(
            [actor.append_splits.remote() for actor in shuffle_actors]
        )
        for ref in sublist
    ]
)
print(f"time to append splits: {time.time() - start}")
ray.get([actor.print_list_of_dfs.remote() for actor in shuffle_actors])
