import numpy as np

def parse_feature_CM(CM_Path):
    CM_vec = np.zeros([133885, 900])
    CM_file = open(CM_Path).read().split('\n')
    for i in CM_file[:-1]:
        qm9_entry = i.split()
        qm9_ind = int(qm9_entry[0].replace("qm9:",""))
        for Indj, j in enumerate(qm9_entry[1:]):
            CM_vec[qm9_ind-1][Indj] = float(j)
    return CM_vec