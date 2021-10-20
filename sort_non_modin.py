from modin.config import NPartitions
from modin.distributed.dataframe.pandas import partitions, unwrap_partitions
from modin.distributed.dataframe.pandas import from_partitions
import numpy as np
import pandas
import ray
import modin.pandas as pd
import time

ray.init(num_cpus=16)
num_partitions = 8
num_actors = 8
NPartitions.put(num_partitions)


@ray.remote(num_cpus=0.001)
class ShuffleActor(object):
    def __init__(self, pos):
        self.list_of_dfs = []
        self.other_actors = None
        self.num_position = pos

    def add_pool(self, other_actors):
        self.other_actors = other_actors

    def combine_and_apply_dfs(self, func):
        full_df = pandas.concat(self.list_of_dfs)
        return func(full_df)

    def append_df(self, df):
        self.list_of_dfs.append(df)

    def split_df(self, df, split_func):
        print(
            f"Actor {self.num_position} split_df: Starting at time: {time.time() - start}"
        )
        [
            self.other_actors[i].append_df.remote(df_split)
            if i != self.num_position
            else self.append_df(df_split)
            for i, df_split in enumerate(split_func(df, self.num_position))
        ]
        print(
            f"Actor {self.num_position} split_df: Finished at time: {time.time() - start}"
        )


def split_func(df, actor_position):
    print(f"Actor {actor_position} split_func: Starting at time: {time.time() - start}")
    df = df.sort_values(columns)
    t = np.digitize(df[columns].squeeze(), quants, right=True)
    grouper = df.groupby(t)
    print(
        f"Actor {actor_position} split_func: finished groupby at time: {time.time() - start}"
    )
    return [
        grouper.get_group(i)
        if i in grouper.keys
        else pandas.DataFrame(columns=df.columns)
        for i in range(len(quants))
    ]


df = pd.read_csv("test_1mx256.csv")
start = time.time()

shuffle_actors = [ShuffleActor.remote(i) for i in range(num_actors)]
parts = unwrap_partitions(df, axis=0)
assert (
    len(parts) == num_partitions
), f"Dataframe was partitioned into {len(parts)} partitions instead of {num_partitions} partitions."
ray.get([actor.add_pool.remote(shuffle_actors) for actor in shuffle_actors])
print(
    f"Set up {len(shuffle_actors)} actors and {len(parts)} partitions at: {time.time() - start}"
)

columns = "col2"
# last quantile is 99.0 but digitize uses <=: https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
# Solution: right bin is max and use right=True so last bin is <= max and first bin is <= 1/N quantile
# so every value is captured.
# Calculating qunatiles is really slow!
sample = df.iloc[
    np.sort(np.random.choice(len(df), size=10 * num_partitions, replace=False))
]._to_pandas()
quants = [
    np.quantile(sample[columns], i / num_actors) for i in range(1, num_actors + 1)
]
quants[-1] = float(
    "inf"
)  # Because of sampling, the max quantile may not be the actual max
print(f"Got quantiles at: {time.time() - start}")

ray.get(
    [
        shuffle_actors[i % num_actors].split_df.remote(partition, split_func)
        for i, partition in enumerate(parts)
    ]
)
print(f"Split dataframes at: {time.time() - start}")

new_parts = [
    actor.combine_and_apply_dfs.remote(lambda x: x.sort_values(columns))
    for actor in shuffle_actors
]
print(f"Combined dataframes and sorted each partition at: {time.time() - start}")

df = from_partitions(new_parts, axis=0)
print(
    f"Built dataframe from {len(new_parts)} parts and finished at: {time.time() - start}"
)
