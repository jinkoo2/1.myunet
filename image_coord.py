import numpy as np
from rect import rect


class image_coord:
    def __init__(self, size=None, origin=None, spacing=None, direction=None):
        self.size = np.array(size).astype(int)
        self.origin = np.array(origin).astype(float)
        self.spacing = np.array(spacing).astype(float)
        if direction is not None:
            self.direction = np.array(direction).astype(float)
        else:
            self.direction = np.array(
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    def to_np_array(self):
        return np.concatenate([self.size.astype(float), self.spacing, self.origin, self.direction])

    def from_np_array(self, arr):
        self.size = np.array([arr[0], arr[1], arr[2]]).astype(int)
        self.origin = np.array([arr[3], arr[4], arr[5]]).astype(float)
        self.spacing = np.array([arr[6], arr[7], arr[8]]).astype(float)
        self.direction = np.array([arr[9], arr[10], arr[11], arr[12],
                                   arr[13], arr[14], arr[15], arr[16], arr[17]]).astype(float)

    def __str__(self):
        return 'size:{self.size},origin:{self.origin},spacing:{self.spacing},direction:{self.direction}'.format(self=self)

    def rect_o(self):
        return rect(self.origin, self.origin+self.size*self.spacing)

    def rect_I(self):
        return rect(np.array([0, 0, 0]).astype(int), self.size)

    # convert a point in w to I.
    def w2I(self, pt_w):
        return np.round((pt_w-self.origin)/self.spacing)

    # convert a point in w to o
    def w2o(self, pt_o):
        return (pt_o-self.origin)

    # convert a point in w to u (normalized coordinate system)
    def w2u(self, pt_o):
        return (pt_o-self.origin)/self.rect_o().size()

    # convert a point in w to I.
    def o2I(self, pt_o):
        return np.round(pt_o/self.spacing)

    # convert a point in w to u (normalized coordinate system)
    def o2u(self, pt_o):
        return pt_o/self.rect_o().size()

    # convert a point in I to o
    def I2w(self, pt_I):
        return (pt_I*self.spacing+self.origin)
