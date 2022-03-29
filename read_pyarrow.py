import pandas
from modin.config import NPartitions
from modin.distributed.dataframe.pandas import from_partitions
from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.config import NPartitions
from pyarrow import csv
import pyarrow as pa
import ray


@ray.remote
def read_block(path: str, column_start: int, column_end: int):
    with pa.memory_map(path, "r") as source:
        table = pa.ipc.RecordBatchFileReader(source).read_all()
    pandas_object = table.select(list(range(column_start, column_end))).to_pandas()
    if isinstance(pandas_object, pandas.Series):
        pandas_object = pandas_object.to_frame()
    return pandas_object


# This reads in 5.02 seconds on my mac
def read_pyarrow(path: str):
    with pa.memory_map(path, "r") as source:
        table = pa.ipc.RecordBatchFileReader(source).read_all()
    num_columns = table.shape[1]
    col_step = compute_chunksize(num_columns, NPartitions.get())
    return from_partitions(
        [
            read_block.remote(
                path,
                col_start,
                col_start + col_step,
            )
            for col_start in range(0, num_columns, col_step)
        ],
        axis=1,
        index=table.select([]).to_pandas().index,
    )


ray.init()
