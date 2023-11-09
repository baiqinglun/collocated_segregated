from enum import Enum

# 面的索引
class FaceId2D(Enum):
    east = 0
    west = 1
    south = 2
    north = 3
    count = 4


class FaceId3D(Enum):
    east = 0
    west = 1
    south = 2
    north = 3
    top = 4
    bottom = 5
    count = 6


# 4种物理边界条件
class PhysicsBoundaryID(Enum):
    none = 0
    wall = 1
    inlet = 2
    outlet = 3
    count = 4

# 温度边界条件
class TemperatureBoundaryID(Enum):
    constant = 0
    heat_flux = 1
    count = 2


# 单元的边界
class BoundaryLimitID(Enum):
    none = 0 # 内部面
    x_min = 1
    x_max = 2
    y_min = 3
    y_max = 4
    z_min = 5
    z_max = 6
    count = 7