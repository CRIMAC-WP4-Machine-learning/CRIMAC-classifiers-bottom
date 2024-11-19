# -*- coding: utf-8 -*-
"""
The main function for bottom detection.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import datetime
import hashlib
import os
from dataclasses import asdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from numcodecs import Blosc

from bottomdetection import bottom_annotation
from bottomdetection import simple_bottom_detector
from bottomdetection import edge_bottom_detector
from bottomdetection import angle_bottom_detector
from bottomdetection import combined_bottom_detector
from bottomdetection import work_files_bottom_detector
from bottomdetection.parameters import Parameters


def run(zarr_file: str,
        out_file: str,
        work_dir: str = '',
        algorithm: str = 'simple',
        parameters: Parameters = Parameters()) -> None:

    print(f"Opening zarr file: {zarr_file}")
    zarr_data = xr.open_zarr(zarr_file, chunks={'frequency': 'auto', 'ping_time': 'auto', 'range': -1})

    print(f'\n\nInput: {zarr_file}')
    print(zarr_data)

    channel_index = 0

    attributes = make_attributes(zarr_file, zarr_data, algorithm, parameters)

    bottom_depth = detect_bottom(zarr_data, work_dir, algorithm, parameters)
    print('\n\nBottom depth:')
    print(bottom_depth)

    _, out_file_extension = os.path.splitext(out_file)
    if out_file_extension in {'.csv', '.html', '.parquet'}:
        bottom_df = bottom_annotation.to_pandas(zarr_data, channel_index, bottom_depth)
        print(f'\n\nOutput: {out_file}')
        print(bottom_df)
        write_pandas(bottom_df, out_file)

    elif out_file_extension in {'.nc', '.zarr'}:
        bottom_ds = bottom_annotation.to_xarray(zarr_data, channel_index, bottom_depth, attributes)
        print(f'\n\nOutput: {out_file}')
        print(bottom_ds)
        write_xarray(bottom_ds, out_file)

    else:
        raise ValueError('Unknown file format: ' + out_file)


def make_attributes(zarr_file: str, zarr_data: xr.Dataset, algorithm: str, parameters: Parameters) -> dict:
    attributes = dict(
        name='CRIMAC-classifiers-bottom',
        version=os.getenv('VERSION_NUMBER', 'Development version'),
        commit_sha=os.getenv('COMMIT_SHA', 'Unknown git_hash'),
        description='Bottom detection',
        time=datetime.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z',
        input_file=zarr_file,
        input_hash='todo: ... ' + zarr_data.attrs['description'],
        algorithm=algorithm,
    )
    for key, value in asdict(parameters).items():
        attributes[f'parameter_{key}'] = value

    hasher = hashlib.sha1()
    for value in attributes.values():
        hasher.update(str(value).encode('UTF-8'))
    attributes['hash'] = hasher.hexdigest()

    return attributes


def detect_bottom(zarr_data: xr.Dataset, work_dir: str, algorithm: str, parameters: Parameters) -> xr.DataArray:
    if algorithm == 'simple':
        return simple_bottom_detector.detect_bottom(zarr_data, parameters)

    if algorithm == 'edge':
        return edge_bottom_detector.detect_bottom(zarr_data, parameters)

    if algorithm == 'angles':
        return angle_bottom_detector.detect_bottom(zarr_data, parameters)

    if algorithm == 'combined':
        return combined_bottom_detector.detect_bottom(zarr_data, parameters)

    if algorithm == 'work_files':
        return work_files_bottom_detector.detect_bottom(zarr_data, work_dir)

    if algorithm.startswith('constant'):
        # A very fast algorithm for testing and debugging.
        depth = float(algorithm[8:]) if len(algorithm) > 8 else 100
        bottom_depth = np.full(len(zarr_data['ping_time']), depth)
        return xr.DataArray(name='bottom_depth', data=bottom_depth, dims=['ping_time'],
                            coords={'ping_time': zarr_data['ping_time']})

    raise ValueError('Unknown bottom algorithm: ' + algorithm)


def write_pandas(df: pd.DataFrame, out_file: str) -> None:
    if out_file.endswith('.csv'):
        df.to_csv(out_file)

    elif out_file.endswith('.html'):
        df.to_html(out_file)

    elif out_file.endswith('.parquet'):
        table = pa.Table.from_pandas(df)
        with pq.ParquetWriter(out_file, table.schema) as writer:
            writer.write_table(table=table)

    else:
        raise ValueError('Unknown file format: ' + out_file)


def write_xarray(ds: xr.Dataset, out_file: str) -> None:
    if out_file.endswith('.nc'):
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(out_file, mode='w', encoding=encoding)

    elif out_file.endswith('.zarr'):
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
        encoding = {var: {'compressor': compressor} for var in ds.data_vars}
        ds.to_zarr(out_file, mode='w', encoding=encoding)

    else:
        raise ValueError('Unknown file format: ' + out_file)
