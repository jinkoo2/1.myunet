import numpy as np

class rect:
    def __init__(self, low, high):
        self.low = np.array(low)
        self.high = np.array(high)

    def __str__(self):
        return 'Low:{self.low},High:{self.high}'.format(self=self)

    def size(self):
        return self.high-self.low

    def center(self):
        return (self.high-self.low)/2

    def intersect(self, rect0):
        new_low = np.maximum(self.low, rect0.low)
        new_high = np.minimum(self.high, rect0.high)
        return rect(new_low, new_high)
    
    def expand(self, num):
        return rect(self.low-[num]*len(self.low), self.high+[num]*len(self.low))
    
# r0 = rect(np.array([0.0]*3), np.array([100.0]*3))
# print(r0)
# print(r0.expand(10))
# r1 = rect(np.array([10.0]*3), np.array([90.0]*3))

# print(r0.low)
# print(r0.high)
# print(r0.width())
# r2 = r0.intersect(r1)
# print(r2.low)
# print(r2.high)



