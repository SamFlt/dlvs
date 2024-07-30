from inference.utils import VSArguments
from typing import Any, Dict, Tuple, List, Generic,  TypeVar, Optional, NoReturn, NewType, Union, Callable
from nptyping import NDArray, Float, UInt
from torch import Tensor

Shape = TypeVar("Shape")
DType = TypeVar("DType")

UnsignedInt = NewType('Unsigned', int) # int >= 0
PositiveFloat = NewType('PositiveFloat', float) # float >= 0
PositiveNonZeroFloat = NewType('PositiveNonZeroFloat', float) # float > 0


RotationMatrix = NDArray[(3, 3), float]
HomogeneousMatrix = NDArray[(4, 4), float]
HomogeneousMatrixArray = NDArray[(Any, 4, 4), float]


GeneratorParameters = Dict[str, Any]
Pose = NDArray[(6,), float]
PoseArray = NDArray[(Any, 6), float]
Velocity = NDArray[(6,), float]
VelocityArray = NDArray[(Any, 6), float]
VelocityTrajectoryArray = NDArray[(Any, Any, 6), Float] # N x T x 6, T is Time, sequence of velocities
Point3DArray = NDArray[(Any, 3), Float]


VSErrorArray = NDArray[(Any, Any), Float] # N x error_dim

RawRGBImageArray = NDArray[(Any, Any, Any, 3), UInt] # N x H x W x 3
RGBImageArrayTorchFormat = NDArray[(Any, 3, Any, Any), Float] # N x 3 x H x W
RawGrayImageArray = NDArray[(Any, Any, Any, 1), UInt] # N x H x W x 1
GrayImageArrayTorchFormat = NDArray[(Any, 1, Any, Any), Float] # N x 1 x H x W 
GrayImageTorchArray = Tensor
RGBImageTorchArray = Tensor
ImageTorchArray = Union[GrayImageTorchArray, RGBImageTorchArray]


LoggingActions = List[Tuple[str, Tuple[Any, ...], bool]] # str = action to save, Tuple[any, ...] = function arguments, bool = should always save (ignore the --save_plots etc, flags)
VSMethodBuilder = Tuple[PositiveNonZeroFloat, str, Callable[[VSArguments], 'VSMethod']]