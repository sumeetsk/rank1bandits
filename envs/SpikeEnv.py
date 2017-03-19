import numpy as np

class SpikeEnv:
    def __init__(self, K = 4, L = 4, baseu = 0.5, gapu = 0.4, basev = 0.5, gapv = 0.4):
        self.K = K
        self.L = L
        
        self.ubar = baseu * np.ones(K)
        self.ubar[K // 2] += gapu
        self.vbar = basev * np.ones(L)
        self.vbar[L // 2] += gapv
        
        self.ut = np.zeros(K)
        self.vt = np.zeros(L)
        
    def num_rows(self):
        return self.K
        
    def num_cols(self):
        return self.L
        
    def randomize(self):
        # randomize row and column variables
        self.ut = np.array(np.random.uniform(size = self.K) < self.ubar, dtype = np.int)
        self.vt = np.array(np.random.uniform(size = self.L) < self.vbar, dtype = np.int)
        
    def reward(self, action):
        # reward of action (row-column pair)
        (i, j) = tuple(action)
        return self.ut[i] * self.vt[j]
        
    def regret(self, action):
        # regret of action (row-column pair)
        (i, j) = tuple(action)
        return self.ut[np.argmax(self.ubar)] * self.vt[np.argmax(self.vbar)] - self.ut[i] * self.vt[j]
        
    def pregret(self, action):
        # pseudo-regret of action (row-column pair)
        (i, j) = tuple(action)
        return np.amax(self.ubar) * np.amax(self.vbar) - self.ubar[i] * self.vbar[j]

    def plot(self):
        # plot row and column probabilities
        fig, (left, right) = plt.subplots(ncols = 2, figsize = (14, 4))
        left.plot(self.ubar)
        right.plot(self.vbar)
