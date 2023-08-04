import numpy as np
import faiss

nq = 100
nb = 1000
d = 32

xq = faiss.randn((nq, d))
xb = faiss.randn((nb, d))

# reference IP search
k = 10
index = faiss.IndexFlatIP(d)
index.add(xb)
Dref, Iref = index.search(xq, k)

# see http://ulrichpaquet.com/Papers/SpeedUp.pdf theorem 5

def get_phi(xb):
    return (xb ** 2).sum(1).max()

def augment_xb(xb, phi=None):
    norms = (xb ** 2).sum(1)
    if phi is None:
        phi = norms.max()
    extracol = np.sqrt(phi - norms)
    return np.hstack((xb, extracol.reshape(-1, 1)))

def augment_xq(xq):
    extracol = np.zeros(len(xq), dtype='float32')
    return np.hstack((xq, extracol.reshape(-1, 1)))

# reference IP search
k = 10
index = faiss.IndexFlatL2(d + 1)
index.add(augment_xb(xb))
D, I = index.search(augment_xq(xq), k)

np.all(I == Iref)
