# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.


"""Collection of utility functions for the PandasDataFrame."""

from modin.logging import get_logger


import pandas
from pandas.api.types import union_categoricals


def concatenate(dfs, for_compute_dtypes: bool = False):
    """
    Concatenate pandas DataFrames with saving 'category' dtype.

    All dataframes' columns must be equal to each other.

    Parameters
    ----------
    dfs : list
        List of pandas DataFrames to concatenate.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame.
    """

    logger = get_logger()
    logger_level = getattr(logger, "info")
    for df in dfs:
        assert df.columns.equals(dfs[0].columns)
    logger_level(f"utils::concatenate: made it past assertion")
    logger_level(f"utils::concatenate: getting first df...")
    first_df = dfs[0]
    logger_level(f"utils::concatenate: got first df")
    logger_level(f"utils::concatenate: getting first df columns...")
    first_df_columns = dfs[0].columns
    logger_level(f"utils::concatenate: got first df columns.")
    logger_level(f"utils::concatenate: getting len of first df columns...")
    first_df_columns_len = len(dfs[0].columns)
    logger_level(f"utils::concatenate: got len of first df columns.")
    logger_level(
        f"utils::concatenate: iterating through all {first_df_columns_len} columns of all {len(dfs)} dfs."
    )
    for i in df.columns.get_indexer_for(df.select_dtypes("category")):
        columns = [df.iloc[:, i] for df in dfs]
        logger_level(f"utils::concatenate: got all columns {i}.")
        union = union_categoricals(columns)
        logger_level(f"utils::concatenate: got union of all columns {i}.")
        for df_index, df in enumerate(dfs):
            logger_level(
                f"utils::concatenate: replacing column {i} in df {df_index}..."
            )
            df.iloc[:, i] = pandas.Categorical(
                df.iloc[:, i], categories=union.categories
            )
            logger_level(f"utils::concatenate: replaced column {i} in df {df_index}...")
    logger_level(f"utils::concatenate: iterated through all columns")
    result = pandas.concat(dfs)
    logger_level(f"utils::concatenate: got concat result")
    return result
