import numpy as np

def parse_prop(Prop_Path):    
    QM9_Prop_File = open(Prop_Path).read().split('\n')[1:-1]
    QM9_Prop = np.zeros([133885, 13])
    QM9_Index = np.zeros(133885)
    for Indi, i in enumerate(QM9_Prop_File):
        temp_strlst = i.split()
        QM9_Index[Indi] = int(temp_strlst[0].replace("qm9:",""))
        for Indj in range(13):
            QM9_Prop[int(temp_strlst[0][4:])-1][Indj] = float(temp_strlst[Indj+2])
    return (QM9_Prop, QM9_Index)