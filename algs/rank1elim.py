import numpy as np
import pdb
import warnings

class Rank1Elim:
    def __init__(self, K, L, T):
        self.K = K
        self.L = L
        self.T = T
        
        self.row_pulls = np.ones((K, L)) # number of pulls in row exploration
        self.row_reward = np.zeros((K, L)) # cumulative reward in row exploration
        self.col_pulls = np.ones((K, L)) # number of pulls in column exploration
        self.col_reward = np.zeros((K, L)) # cumulative reward in column exploration
        
        self.active_rows = np.arange(K)
        self.active_cols = np.arange(L)
        self.row_defeat = np.arange(K)
        self.col_defeat = np.arange(L)
        
        self.EXPLORE_ROWS = 0
        self.EXPLORE_COLS = 1
        self.gap = 1.
        self.budget = np.ceil(2 * np.log(self.T * (self.gap ** 2)) / (self.gap ** 2))
        self.bandit = self.EXPLORE_ROWS
        self.active = 0
        self.i = self.active_rows[self.active]
        self.j = self.col_defeat[np.random.randint(0, L)]
        
    def update(self, t, action, r):
        K = self.K
        L = self.L
        
        # exploration step
        if (self.bandit == self.EXPLORE_ROWS):
            # update rows
            self.row_pulls[self.i, self.j] += 1
            self.row_reward[self.i, self.j] += r
            
            if self.active < self.active_rows.size - 1:
                # continue in row exploration
                self.active += 1
                self.i = self.active_rows[self.active]
                self.j = self.col_defeat[np.random.randint(0, L)]
            else:
                # switch to column exploration
                self.bandit = self.EXPLORE_COLS
                self.active = 0
                self.i = self.row_defeat[np.random.randint(0, K)]
                self.j = self.active_cols[self.active]
        else:
            # update columns
            self.col_pulls[self.i, self.j] += 1
            self.col_reward[self.i, self.j] += r
            
            if self.active < self.active_cols.size - 1:
                # continue in column exploration
                self.active += 1
                self.i = self.row_defeat[np.random.randint(0, K)]
                self.j = self.active_cols[self.active]
            else:
                # switch to row exploration
                self.budget -= 1
                self.bandit = self.EXPLORE_ROWS
                self.active = 0
                self.i = self.active_rows[self.active]
                self.j = self.col_defeat[np.random.randint(0, L)]
        
        # elimination step
        if self.budget == 0:
            c = np.sqrt(np.log(self.T))
            
            # LCB and UCB on the mean payoff of each row
            ndx = self.active_rows
            lcbr = np.sum(self.row_reward[ndx, :], axis = 1) / np.sum(self.row_pulls[ndx, :], axis = 1) - \
                c * np.sqrt(1 / np.sum(self.row_pulls[ndx, :], axis = 1))
            ucbr = np.sum(self.row_reward[ndx, :], axis = 1) / np.sum(self.row_pulls[ndx, :], axis = 1) + \
                c * np.sqrt(1 / np.sum(self.row_pulls[ndx, :], axis = 1))
            
            # determine remaining rows
            bestr = np.argmax(lcbr)
            selr = np.nonzero(lcbr[bestr] <= ucbr)[0]
            remr = np.nonzero(lcbr[bestr] > ucbr)[0]
            
            # LCB and UCB on the mean payoff of each column
            ndx = self.active_cols
            lcbc = np.sum(self.col_reward[:, ndx], axis = 0) / np.sum(self.col_pulls[:, ndx], axis = 0) - \
                c * np.sqrt(1 / np.sum(self.col_pulls[:, ndx], axis = 0))
            ucbc = np.sum(self.col_reward[:, ndx], axis = 0) / np.sum(self.col_pulls[:, ndx], axis = 0) + \
                c * np.sqrt(1 / np.sum(self.col_pulls[:, ndx], axis = 0))
            
            # determine remaining columns
            bestc = np.argmax(lcbc)
            selc = np.nonzero(lcbc[bestc] <= ucbc)[0]
            remc = np.nonzero(lcbc[bestc] > ucbc)[0]
            
            # update data structures
            if (remr.size > 0):
                for i in range(K):
                    if self.row_defeat[i] in self.active_rows[remr]:
                        self.row_defeat[i] = self.active_rows[bestr]
                self.active_rows = self.active_rows[selr]
            if (remc.size > 0):
                for j in range(L):
                    if self.col_defeat[j] in self.active_cols[remc]:
                        self.col_defeat[j] = self.active_cols[bestc]
                self.active_cols = self.active_cols[selc]
            
            old_gap = self.gap
            self.gap /= 2.
            self.budget = np.ceil(4 * np.log(self.T) / (self.gap ** 2)) - \
                np.ceil(4 * np.log(self.T) / (old_gap ** 2))
            self.bandit = self.EXPLORE_ROWS
            self.active = 0
            self.i = self.active_rows[self.active]
            self.j = self.col_defeat[np.random.randint(0, L)]
            
#             print("Time:", t, "\nGap:", self.gap, "\nRows:", self.row_index, "\nColumns:", self.col_index)
            
    def get_action(self, t):
        return (self.i, self.j)
