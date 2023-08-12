import numpy as np
from math import pi


TABLE_ORIGIN = np.array([0, -0.65, 1.115])
def table2world(position):
    return position + TABLE_ORIGIN


OBJECT_START_POSITION = table2world(np.array([0, 0, 0.25]))  # just a default
OBJECT_START_ORIENTATION = [0, 0, 0]  # just a default orientation

PRE_GRASP_DISTANCE = 0.15

PICKUP_POINT = table2world(np.array([0, 0, 0.4]))
PICKUP_ORIENTATION = [0, pi/4, 0]

MIN_HEIGHT_DIFF_FOR_SUCCESS = 0.10