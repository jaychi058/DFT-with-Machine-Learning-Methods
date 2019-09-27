import numpy as np
import time

# Function import
import feature
import prop
import fold
import fitRF

fOut = open("CM-RF.txt", "w")

#  1. CM vector parsing
CM_Path = "/Volumes/Macintosh HD 2/FSU/2018Fall/SeminarJose/Seminar2/FinalProject/Supplementary_Materials/CM"
time0 = time.time()
CM_vec = feature.parse_feature_CM(CM_Path)
# print(CM_vec[1])
time1 = time.time()
fOut.write("Execuation time 01 - CM vector parsing: %s sec.\n" % (time1 - time0))

#  2. Property parsing
QM9_Prop, QM9_Index = prop.parse_prop("/Volumes/Macintosh HD 2/FSU/2018Fall/SeminarJose/Seminar2/FinalProject/Supplementary_Materials/qm9-mol-info-standardized-v1")
time2 = time.time()
fOut.write("Execuation time 02 - Property parsing: %s sec.\n" % (time2 - time1))

#  3. Fold parsing
Predefined_Split, Train_Index, Test_Index = fold.predefined_fold()
time3 = time.time()
fOut.write("Execuation time 03 - Fold parsing: %s sec. \n" % (time3 - time2))

#  4. Reindex to training and testing set
Train_X = np.zeros([len(Train_Index[0]), CM_vec.shape[1]])
Train_Y = np.zeros([13, len(Train_Index[0])])
Test_X = np.zeros([len(Test_Index[0]), CM_vec.shape[1]])
Test_Y = np.zeros([13, len(Test_Index[0])])
Predict_Y = np.zeros([13, len(Test_Index[0])])
for Indi, i in enumerate(Train_Index[0]):
    Train_X[Indi] = CM_vec[i-1]
    Train_Y[:,Indi] = QM9_Prop[i-1,:]
for Indi, i in enumerate(Test_Index[0]):
    Test_X[Indi] = CM_vec[i-1]
    Test_Y[:,Indi] = QM9_Prop[i-1,:]
time4 = time.time()
fOut.write("Execuation time 04 - Reindex: %s sec. \n" % (time4 - time3))

#  5. Training and Testing - Random forest (RF)
test_PATH = "/Volumes/Macintosh HD 2/FSU/2018Fall/SeminarJose/Seminar2/FinalProject/Supplementary_Materials/qm9-prop-stats-v1"
test_file = open(test_PATH).read().split('\n')
stdevp_vec = np.zeros(13)
MAE = np.zeros(13)
for i in test_file[2:-1]:
    tempStdevp = i.split()
    for j in range (1, np.size(tempStdevp)):
        stdevp_vec[j-1] = float(tempStdevp[j])
#print tempStdevp
#print stdevp_vec

fOut.write("\n")
fOut.write("--- Training and Testing RF ---\n")
fOut.write("\n")
fOut.write(" Prop_Id" + "      Err_MAE" + "      Time_Train" + "  Time_Test" + "\n" )
for Indi in range(0, 1):
    (Err_MAD, Err_RMSD, Time_Train, Time_Test) = \
        fitRF.fit_RF(Train_X, Train_Y[Indi], Test_X, Test_Y[Indi], Predefined_Split[0])
    MAE[Indi] = Err_MAD*stdevp_vec[Indi]
    fOut.write("{:8}{:14.8f}{:11.2f}{:11.2f}\n".format(Indi, MAE[Indi], Time_Train, Time_Test))
fOut.close()