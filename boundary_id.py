'''边界条件的枚举类型'''
from enum import Enum

class FaceId2D(Enum):
    '''面的索引(2D)'''
    EAST = 0
    WEST = 1
    SOUTH = 2
    NORTH = 3
    COUNT = 4


class FaceId3D(Enum):
    '''面的索引(3D)'''
    EAST = 0
    WEST = 1
    SOUTH = 2
    NORTH = 3
    TOP = 4
    BOTTOM = 5
    COUNT = 6

class PhysicsBoundaryID(Enum):
    '''物理边界条件'''
    NONE = 0
    WALL = 1
    INLET = 2
    OUTLET = 3
    COUNT = 4

class VelocityBoundaryID(Enum):
    '''速度边界条件'''
    NONE = 0
    CONSTANT = 1
    VELOCITY_FLUX = 2
    COUNT = 3

class TemperatureBoundaryID(Enum):
    '''温度边界条件'''
    NONE = 0
    CONSTANT = 1
    HEAT_FLUX = 2
    COUNT = 3

class BoundaryLimitID(Enum):
    '''单元的边界'''
    NONE = 0 # 内部面
    X_MIN = 1
    X_MAX = 2
    Y_MIN = 3
    Y_MAX = 4
    Z_MIN = 5
    Z_MAX = 6
    COUNT = 7
