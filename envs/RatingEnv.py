import numpy as np
from SpikeEnv import SpikeEnv

class RatingEnv(SpikeEnv):
    def __init__(self, attrfile, examfile):
        self.ubar = np.genfromtxt(attrfile)
        self.vbar = np.genfromtxt(examfile)
        self.K = len(self.ubar)
        self.L = len(self.vbar)

        self.ut = np.zeros(self.K)
        self.vt = np.zeros(self.L)
