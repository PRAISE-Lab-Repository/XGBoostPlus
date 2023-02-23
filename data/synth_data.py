import numpy as np
import pandas as pd

# experiment 1: noiseless labels as privileged info
def synthetic_01(a,n):
    x  = np.random.randn(n,a.size)
    e  = (np.random.randn(n))[:,np.newaxis]
    xs = np.dot(x,a)[:,np.newaxis]
    y  = ((xs+e) > 0).ravel()
    return (xs,x,y)

# experiment 2: noiseless inputs as privileged info (violates causal assump)
def synthetic_02(a,n):
    x  = np.random.randn(n,a.size)
    e  = np.random.randn(n,a.size)
    y  = (np.dot(x,a) > 0).ravel()
    xs = np.copy(x)
    x  = x+e
    return (xs,x,y)

# experiment 3: relevant inputs as privileged info
def synthetic_03(a,n):
    x  = np.random.randn(n,a.size)
    xs = np.copy(x)
    xs = xs[:,0:3]
    a  = a[0:3]
    y  = (np.dot(xs,a) > 0).ravel()
    return (xs,x,y)

# experiment 4: sample dependent relevant inputs as privileged info
def synthetic_04(a,n):
    x  = np.random.randn(n,a.size)
    xs = np.copy(x)
    #xs = np.sort(xs,axis=1)[:,::-1][:,0:3]
    xs = xs[:,np.random.permutation(a.size)[0:3]]
    a  = a[0:3]
    tt = np.median(np.dot(xs,a))
    y  = (np.dot(xs,a) > tt).ravel()
    return (xs,x,y)