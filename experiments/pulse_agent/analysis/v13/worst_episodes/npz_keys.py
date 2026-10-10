import sys, numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
for k in d.files:
    a = d[k]
    print(k, a.shape, a.dtype)
