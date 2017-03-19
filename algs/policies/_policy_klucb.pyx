cimport cython
import numpy as np
cimport numpy as np
import sys

from libc.math cimport log as c_log
from libc.math cimport abs as c_abs
from libc.stdio cimport printf

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT

np.import_array()

cdef double DELTA = 1e-12
cdef double GAMMA = 1e-3

cdef inline double double_max(double a, double b): return a if a >= b else b
cdef inline double double_min(double a, double b): return a if a <= b else b


cdef double KLBernoulli(double p, double q):
    if p > 1 or q > 1:
        return -1
    if p == 0:
        return - c_log(1 - q)
    return p * c_log(p / q) + (1 - p) * c_log((1 - p) / (1 - q))

cpdef np.ndarray[DOUBLE, ndim=1] _computeKLUCB(
        int n,
        np.ndarray[DOUBLE, ndim=1] gains,
        np.ndarray[DOUBLE, ndim=1] N_plays,
        int n_iter):
    cdef:
        Py_ssize_t i
        Py_ssize_t j
        double upper_bound = 0
        double u_m = 0
        double u_M = 0
        double f_max = 0
        double p = 0
        cdef Py_ssize_t K = gains.shape[0]
        np.ndarray[DOUBLE, ndim=1] res = np.zeros(
                K, dtype=np.float64)
    for i in range(K):
        upper_bound = (c_log(n) + c_log( c_log(n) ) ) / N_plays[i]
        #(1 + GAMMA) * c_log(n) / N_plays[i]
        p = gains[i] / N_plays[i]
        if p == 0:
            res[i] = 1
            continue
        u_m = p
        u_M = 1 - DELTA
        f_max = upper_bound - KLBernoulli(p, u_M)
        if u_m >= 1 or f_max >= 0:
            res[i] = 1
            continue
        j = 0
        for j in range(n_iter):
            if (upper_bound - KLBernoulli(p, (u_m + u_M) / 2)) >= 0:
                u_m = (u_m + u_M) / 2
            else:
                u_M = (u_m + u_M) / 2
        res[i] = (u_m + u_M) / 2
    return res


cpdef np.ndarray[DOUBLE, ndim=1] _computeKLLCB(
        int n,
        np.ndarray[DOUBLE, ndim=1] gains,
        np.ndarray[DOUBLE, ndim=1] N_plays,
        int n_iter):
    cdef:
        Py_ssize_t i
        Py_ssize_t j
        double upper_bound = 0
        double u_m = 0
        double u_M = 0
        double f_max = 0
        double p = 0
        cdef Py_ssize_t K = gains.shape[0]
        np.ndarray[DOUBLE, ndim=1] res = np.zeros(
                K, dtype=np.float64)
    for i in range(K):
        upper_bound = (c_log(n) + c_log( c_log(n) ) ) / N_plays[i]
        #upper_bound = 2 * c_log(n) / N_plays[i]
        #(1 + GAMMA) * c_log(n) / N_plays[i]
        p = gains[i] / N_plays[i]
        if p == 0:
            res[i] = 0
            continue
        u_M = p
        u_m = DELTA
        f_max = upper_bound - KLBernoulli(p, u_m)
        if  f_max >= 0:
            res[i] = 0
            continue
        j = 0
        for j in range(n_iter):
            if (upper_bound - KLBernoulli(p, (u_m + u_M) / 2)) >= 0:
                u_M = (u_m + u_M) / 2
            else:
                u_m = (u_m + u_M) / 2
        res[i] = (u_m + u_M) / 2
    return res
