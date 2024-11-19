# -*- coding: utf-8 -*-
"""
The entry point when running in a Docker container.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import os
import shutil
import sys
from dataclasses import asdict

import dask
from dask.distributed import Client

from bottomdetection import bottom_detection_main
from bottomdetection.parameters import Parameters

if __name__ == '__main__':
    if os.getenv('DEBUG', 'false') == 'true':
        print('Press enter...')
        input()
        sys.exit(0)

    input_name = os.getenv('INPUT_NAME', '.')
    output_name = os.getenv('OUTPUT_NAME', 'out.parquet')
    algorithm = os.getenv('ALGORITHM', 'simple')

    in_dir = os.path.expanduser('/datain')
    out_dir = os.path.expanduser('/dataout')
    work_dir = os.path.expanduser('/workin')

    parameters = Parameters()
    for key in asdict(parameters).keys():
        string_value = os.getenv(f'PARAMETER_{key}')
        if not string_value:
            continue
        value = float(string_value)
        setattr(parameters, key, value)

    # Setting dask
    tmp_dir = os.path.expanduser(out_dir + '/tmp')

    dask.config.set({'temporary_directory': tmp_dir})
    client = Client()
    print(client)

    print(f"Zarr file: {in_dir}/{input_name}")
    print(f"Out file: {out_dir}/{output_name}")
    print(f"Work dir: {work_dir}")
    print(f"Algorithn: {algorithm}")
    print(f"Parameters: {parameters}")

    bottom_detection_main.run(zarr_file=in_dir + '/' + input_name,
                              out_file=out_dir + '/' + output_name,
                              work_dir=work_dir,
                              algorithm=algorithm,
                              parameters=parameters)

    # Cleaning up
    client.close()
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
