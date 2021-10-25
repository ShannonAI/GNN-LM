# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from pyarrow.plasma import ObjectID
import pyarrow.plasma as plasma
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)


class PlasmaArray(object):
    """
    Wrapper around numpy arrays that automatically moves the data to shared
    memory upon serialization. This is particularly helpful when passing numpy
    arrays through multiprocessing, so that data is not unnecessarily
    duplicated or pickled.
    Args:
        array: numpy array. if array is None, the array is retrieved from plasma server
        path: plasma client path
        obj_id: if is not None, try to find array in existed PlasmaClient
    """

    def __init__(self, array: np.array = None, path: str = "/tmp/plasma", obj_id: ObjectID = None):
        super().__init__()
        self.array = array
        self.object_id = obj_id
        self.path = path

        # variables with underscores shouldn't be pickled
        self._client = None
        self._server = None
        self._server_tmp = None

        if self.array is None:
            assert obj_id is not None, "must provide array or obj_id"
            self.array = self.client.get(obj_id)

        self.disable = self.object_id is None and self.array.nbytes < 134217728  # disable for arrays <128MB

        if obj_id is None:
            ngb = self.array.nbytes // 1024 ** 3
            if ngb > 1:
                logger.info(f"new array have size of {ngb}G")

    @staticmethod
    def generate_object_id(path: str, encoding="utf-8", suffix="") -> ObjectID:
        """generate object id used by plasma in-memory store from file path"""
        path += suffix
        truncate = path[-20:]
        if len(truncate) < 20:
            truncate = "a" * (20 - len(truncate)) + truncate
        return ObjectID(bytes(truncate, encoding=encoding))

    def start_server(self):
        assert self.object_id is None
        self._server_tmp = tempfile.NamedTemporaryFile()
        self.path = self._server_tmp.name
        self._server = subprocess.Popen(
            [
                "plasma_store",
                "-m",
                str(int(1.05 * self.array.nbytes)),
                "-s",
                self.path,
            ]
        )

    @property
    def client(self) -> plasma.PlasmaClient:
        if self._client is None:
            self._client = plasma.connect(self.path)
        return self._client

    def disconnect(self):
        if self._client is not None:
            self._client.disconnect()
        self._client = None

    def __getstate__(self):
        if self.object_id is None:
            if self.disable:
                return self.__dict__
            self.start_server()
            self.object_id = self.client.put(self.array)

        state = self.__dict__.copy()

        del state["array"]
        state["_client"] = None
        state["_server"] = None
        state["_server_tmp"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.disable:
            return
        self.array = self.client.get(self.object_id)
        self.disconnect()

    def __del__(self):
        self.disconnect()
        if self._server is not None:
            self._server.kill()
            self._server = None
            self._server_tmp.close()
            self._server_tmp = None

    def __getitem__(self, item):
        return self.array[item]

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def shape(self):
        return self.array.shape