
num_splits = 16

from modin.distributed.dataframe.pandas import unwrap_partitions
from modin.distributed.dataframe.pandas import from_partitions
import numpy as np
import pandas
import ray
import modin.pandas as pd
import time

ray.init()


@ray.remote(num_cpus=0)
class ShuffleActor(object):

    def __init__(self, pos):
        self.list_of_dfs = []
        self.other_actors = None
        self.num_position = pos

    def add_pool(self, other_actors):
        self.other_actors = other_actors

    def combine_and_apply_dfs(self, func):
        print(f"Self position: {self.num_position}, Length: {len(self.list_of_dfs)}")
        full_df = pandas.concat(self.list_of_dfs)
        return func(full_df)

    def append_df(self, df):
        self.list_of_dfs.append(df)

    def split_df(self, df, split_func):
        [self.other_actors[i].append_df.remote(df_split)
            if i != self.num_position
            else self.append_df(df_split) for i, df_split in enumerate(split_func(df))
        ]

df = pd.read_csv("test_1mx256.csv")
shuffle_actors = [ShuffleActor.remote(i) for i in range(num_splits)]
parts = unwrap_partitions(df, axis=0)
ray.get([actor.add_pool.remote(shuffle_actors) for actor in shuffle_actors])

start = time.time()
columns = "col2"
quants = [np.quantile(df[columns], i / num_splits) for i in range(1, num_splits + 1)]

def split_func(df):
    df = df.sort_values(columns)
    t = np.digitize(df[columns].squeeze(), quants)
    grouper = df.groupby(t)
    return [grouper.get_group(i) if i in grouper.keys else pandas.DataFrame(columns=df.columns) for i in range(len(quants))]

ray.get([actor.split_df.remote(partition, split_func) for actor, partition in zip(shuffle_actors, parts)])
new_parts = [actor.combine_and_apply_dfs.remote(lambda x: x.sort_values(columns)) for actor in shuffle_actors]
df = from_partitions(new_parts, axis=0)
print(f'time elapsed: {time.time() - start}')

