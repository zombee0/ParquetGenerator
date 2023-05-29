import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fastparquet as fp

def generate_floats(n, pct_null, repeats=1):
    nunique = int(n / repeats)
    unique_values = np.random.randn(nunique)

    num_nulls = int(nunique * pct_null)
    null_indices = np.random.choice(nunique, size=num_nulls, replace=False)
    unique_values[null_indices] = np.nan

    return unique_values.repeat(repeats)

DATA_GENERATORS = {
    'float64': generate_floats
}

def generate_data(total_size, ncols, pct_null=0.1, repeats=1, dtype='float64'):
    type_ = np.dtype('float64')
    nrows = total_size / ncols / np.dtype(type_).itemsize

    datagen_func = DATA_GENERATORS[dtype]

    data = {
        'c' + str(i): datagen_func(nrows, pct_null, repeats)
        for i in range(ncols)
    }
    return pd.DataFrame(data)

def write_to_parquet(df, out_path, compression='SNAPPY'):
    arrow_table = pa.Table.from_pandas(df)
    if compression == 'UNCOMPRESSED':
        compression = None
    pq.write_table(arrow_table, out_path, use_dictionary=False,
                   compression=compression)

def read_fastparquet(path):
    return fp.ParquetFile(path).to_pandas()

def read_pyarrow(path, nthreads=1):
    return pq.read_table(path, nthreads=nthreads).to_pandas()

MEGABYTE = 1 << 20
DATA_SIZE = 8 * 1024 * MEGABYTE
NCOLS = 4000

cases = {
    # 'high_entropy': {
    #     'pct_null': 0.1,
    #     'repeats': 1
    # },
    'low_entropy': {
        'pct_null': 0.1,
        'repeats': 1000
    }
}

def get_timing(f, path, niter):
    start = time.clock_gettime(time.CLOCK_MONOTONIC)
    for i in range(niter):
        f(path)
    elapsed = time.clock_gettime(time.CLOCK_MONOTONIC) - start
    return elapsed

NITER = 5

results = []

readers = [
    ('fastparquet', lambda path: read_fastparquet(path)),
    ('pyarrow', lambda path: read_pyarrow(path)),
]

case_files = {}

for case, params in cases.items():
    for compression in ['SNAPPY']:
        path = '{0}_{1}.parquet'.format(case, compression)
        df = generate_data(DATA_SIZE, NCOLS, **params)
        write_to_parquet(df, path, compression=compression)
        df = None
        case_files[case, compression] = path

# for case, params in cases.items():
#     for compression in ['UNCOMPRESSED', 'SNAPPY', 'GZIP']:
#         path = case_files[case, compression]

#         # prime the file cache
#         read_pyarrow(path)
#         read_pyarrow(path)

#         for reader_name, f in readers:
#             elapsed = get_timing(f, path, NITER) / NITER
#             result = case, compression, reader_name, elapsed
#             print(result)
#             results.append(result)
