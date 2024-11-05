import os
from typing import Union

from nptyping import Float, NDArray, Shape

DatasetPathLike = Union[str, os.PathLike]

FloatXx3 = NDArray[Shape["*, 3"], Float]
Float4x4 = NDArray[Shape["4, 4"], Float]

PointCloudXx3 = FloatXx3
Transform4x4 = Float4x4
