import asyncio
from re import split
from modin.config import NPartitions
from modin.distributed.dataframe.pandas import partitions, unwrap_partitions
from modin.distributed.dataframe.pandas import from_partitions
import numpy as np
import pandas
import ray
import modin.pandas as pd
import os
import time

ray.init(num_cpus=8)
num_partitions = 8
num_actors = 8
NPartitions.put(num_partitions)


@ray.remote
class ShuffleActor(object):
    def __init__(self, pos):
        self.list_of_dfs = []
        self.other_actors = None
        self.num_position = pos

    def add_pool(self, other_actors):
        self.other_actors = other_actors

    def combine_and_apply_dfs(self, func):
        full_df = pandas.concat(
            pandas.read_pickle(f"/tmp/shuffle/splits/{self.num_position}/{i}")
            for i in range(num_actors)
        )
        return func(full_df)

    def append_df(self, df, from_actor):
        print(f"appending one df from actor {from_actor} to actor {self.num_position}")
        if type(df) != type(0):
            self.list_of_dfs.append(df)
        print(
            f"done appending one df from actor {from_actor} to actor {self.num_position}"
        )
        return

    def split_df(self, df, split_func, columns, quants, partition_index):
        print(
            f"Actor {self.num_position} split_df on partition {self.num_position}: Starting at time: {time.time() - start}"
        )
        df_path = f"/tmp/shuffle/{self.num_position}"
        df = pandas.read_pickle(df_path)
        print(
            f"Actor {self.num_position} split_df on partition {self.num_position}: has read df at time: {time.time() - start}"
        )
        splits = split_func(df, self.num_position, columns, quants, partition_index)
        for i, split in enumerate(splits):
            split.to_pickle(f"/tmp/shuffle/splits/{i}/{self.num_position}")
        # self.append_df(splits[self.num_position], self.num_position)
        # refs = [
        #     self.other_actors[i].append_df.remote(df, self.num_position)
        #     for i, df_split in enumerate(splits)
        #     if i != self.num_position
        # ]
        print(
            f"Actor {self.num_position} split_df on partition {self.num_position}: Finished at time: {time.time() - start}"
        )
        return 0


def split_func(df, actor_position, columns, quants, partition_index):
    print(
        f"Actor {actor_position} split_func on partition {partition_index}: Starting at time: {time.time() - start}"
    )
    df = df.sort_values(columns)
    t = np.digitize(df[columns].squeeze(), quants, right=True)
    grouper = df.groupby(t)
    splits = [
        grouper.get_group(i)
        if i in grouper.keys
        else pandas.DataFrame(columns=df.columns)
        for i in range(len(quants))
    ]
    print(
        f"Actor {actor_position} split_func on partition {partition_index}: finished at time: {time.time() - start}"
    )
    return splits


start = time.time()


def sorted_dataframe():
    df = pd.read_csv("test_1mx256.csv")

    shuffle_actors = [ShuffleActor.remote(i) for i in range(num_actors)]
    parts = ray.get(unwrap_partitions(df, axis=0))
    print(f"Got partitions at: {time.time() - start}")
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
    # quants = [100 * x / num_actors for x in range(1, num_actors + 1)]
    quants[-1] = float(
        "inf"
    )  # Because of sampling, the max quantile may not be the actual max
    print(f"Got quantiles at: {time.time() - start}")

    for i, partition in enumerate(parts):
        partition.to_pickle(f"/tmp/shuffle/{i}")
    appender_lists = ray.get(
        [
            shuffle_actors[i % num_actors].split_df.remote(
                0, split_func, columns, quants, i
            )
            for i, partition in enumerate(parts)
        ]
    )
    print(f"Got appenders at: {time.time() - start}")
    # ray.get([ref for sublist in appender_lists for ref in sublist])
    print(f"split dataframes at: {time.time() - start}")

    new_parts = [
        actor.combine_and_apply_dfs.remote(
            lambda x: x.sort_values(columns),
        )
        for actor in shuffle_actors
    ]
    ray.get(new_parts)
    print(f"Combined dataframes and sorted each partition at: {time.time() - start}")

    df = from_partitions(new_parts, axis=0)
    print(
        f"Built dataframe from {len(new_parts)} parts and finished at: {time.time() - start}"
    )

    return df


df = sorted_dataframe()
assert len(df) == 1048576
