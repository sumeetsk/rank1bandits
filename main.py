from __future__ import absolute_import, division, print_function
from algs.rank1elimKL import Rank1ElimKL
import numpy as np
import copy
from envs.SpikeEnv import SpikeEnv

def testrun(bandit, env, numExps, T, display = True):
    if display:
        print("Simulation with K = %s and L = %s" % (env.num_rows(), env.num_cols()))

    regret = np.zeros((T // 100, numExps))
    tarray = np.zeros(T//100)
    for ex in range(numExps):
        bandit1 = copy.deepcopy(bandit)

        for t in range(T):
            # generate state
            env.randomize()

            # solve
            (i, j) = bandit1.get_action(t)

            # update model and regret
            bandit1.update(t, (i, j), env.reward((i, j)))
            regret[t // 100, ex] += env.regret((i, j))
            tarray[t//100] = t

        if display:
            regretT = np.sum(regret, axis = 0)
            regretT = regretT[0 : ex + 1]
            print("%.2f \\pm %.2f, " % (np.mean(regretT), np.std(regretT) / np.sqrt(ex + 1)), end = "")
            print()

    return (tarray, regret)

if __name__ == "__main__":
    [baseu, basev, gapu, gapv] = [0.25,0.25,.5,.5]
    numExps = 5
    T = 2000000
    K = 128
    L = K
    s = SpikeEnv(K=K, L=L, baseu=baseu, basev=basev, gapu=gapu, gapv=gapv)
    b = Rank1ElimKL(K=K, L=L, T=T)
    (tarray, regret) = testrun(b, env=s, numExps=numExps, T=T, display=True)
    filename = 'regret'
    np.savez(file=filename, tarray=tarray, regret=regret )
