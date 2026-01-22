# -*- coding: utf-8 -*-



import numpy as np
from multiprocessing import Pool, cpu_count, shared_memory


def init_worker(shared_specs, params):
    """ Initialize workers, attach to shared memory buffers & stores parameter dict.
    """
    global _shared_arrays, _params
    _shared_arrays = {}
    _params = params

    # attach each shared array
    for name, (shm_name, shape, dtype) in shared_specs.items():
        shm = shared_memory.SharedMemory(name=shm_name)
        _shared_arrays[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)