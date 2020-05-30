import pandas as pd
import numpy as np

Max_od = 634.0
num_stamp = 4

# input POI information
a = pd.read_csv("../Data/POI.csv", header=None)
POI_mat = np.array(a)

# input O
Otrain = pd.read_csv('./predictOCNN_train.csv', header=None)
Otrain = np.array(Otrain.values)
# input D
Dtrain = pd.read_csv('./predictDCNN_train.csv', header=None)
Dtrain = np.array(Dtrain.values)

whole_mat = []
T = 42 - num_stamp
AM_num = 11 - num_stamp

for m in range(6):
    # morning peak
    OtrainAM = Otrain[0 + T * m:AM_num + T * m]
    DtrainAM = Dtrain[0 + T * m:AM_num + T * m]
    depth = POI_mat.shape[1] * 2 + 2
    for k in range(AM_num):
        MAT = np.zeros([depth, 285, 285])
        for i in range(POI_mat.shape[1]):
            for j in range(285):
                MAT[0, :, j] = OtrainAM[k, :]
                MAT[1, j, :] = DtrainAM[k, :]
                MAT[i + 2, :, j] = POI_mat[:, i]
                MAT[POI_mat.shape[1] + 2 + i, j, :] = POI_mat[:, i]
        MAT[MAT < 0] = 0
        lst = [0.1937, 0.1906, 0.0641, -0.0076, -0.0148, -0.0216, -0.0084, -0.0112, -0.045, 0.0878, -0.0497, 0.0237,
               -0.0235, 0, -0.0610, -0.0673]
        for i in range(16):
            if (i == 0):
                OD_mat = pow(MAT[i] + 1, lst[i])
            else:
                OD_mat = OD_mat * pow(MAT[i] + 1, lst[i])
        OD_mat = OD_mat * pow(np.e, 0.8110) - 1
        OD_mat = OD_mat / Max_od
        OD_mat = OD_mat[:, :, np.newaxis]
        whole_mat.append(OD_mat)

    # off peak
    OtrainAM = Otrain[11 - num_stamp + T * m:35 - num_stamp + T * m]
    DtrainAM = Dtrain[11 - num_stamp + T * m:35 - num_stamp + T * m]
    depth = POI_mat.shape[1] * 2 + 2
    for k in range(24):
        MAT = np.zeros([depth, 285, 285])
        for i in range(POI_mat.shape[1]):
            for j in range(285):
                MAT[0, :, j] = OtrainAM[k, :]
                MAT[1, j, :] = DtrainAM[k, :]
                MAT[i + 2, :, j] = POI_mat[:, i]
                MAT[POI_mat.shape[1] + 2 + i, j, :] = POI_mat[:, i]
        MAT[MAT < 0] = 0
        lst = [0.0967, 0.1285, 0.0288, -0.0158, 0.0197, -0.0207, 0, -0.0386, -0.0320, 0.0188, -0.0358, 0.0152, -0.0042,
               0.0110, -0.0270, -0.0059]
        for i in range(16):
            if (i == 0):
                OD_mat = pow(MAT[i] + 1, lst[i])
            else:
                OD_mat = OD_mat * pow(MAT[i] + 1, lst[i])
        OD_mat = OD_mat * pow(np.e, 1.7581) - 1
        OD_mat = OD_mat / Max_od
        OD_mat = OD_mat[:, :, np.newaxis]
        whole_mat.append(OD_mat)

    # evening peak
    OtrainAM = Otrain[35 - num_stamp + T * m:42 - num_stamp + T * m]
    DtrainAM = Dtrain[35 - num_stamp + T * m:42 - num_stamp + T * m]
    depth = POI_mat.shape[1] * 2 + 2
    for k in range(7):
        MAT = np.zeros([depth, 285, 285])
        for i in range(POI_mat.shape[1]):
            for j in range(285):
                MAT[0, :, j] = OtrainAM[k, :]
                MAT[1, j, :] = DtrainAM[k, :]
                MAT[i + 2, :, j] = POI_mat[:, i]
                MAT[POI_mat.shape[1] + 2 + i, j, :] = POI_mat[:, i]
        MAT[MAT < 0] = 0
        lst = [0.1771, 0.1822, 0.0905, -0.0789, 0.0395, -0.0309, 0.0328, -0.0727, -0.0717, 0.0552, -0.0078, 0, -0.0235,
               0, -0.0182, -0.0516]
        for i in range(16):
            if (i == 0):
                OD_mat = pow(MAT[i] + 1, lst[i])
            else:
                OD_mat = OD_mat * pow(MAT[i] + 1, lst[i])
        OD_mat = OD_mat * pow(np.e, 1.1103) - 1
        OD_mat = OD_mat / Max_od
        OD_mat = OD_mat[:, :, np.newaxis]
        whole_mat.append(OD_mat)

OD_train = np.array(whole_mat)
np.save('OD_train.npy', OD_train)

Ovalid = pd.read_csv('./predictOCNN_valid.csv', header=None)
Ovalid = np.array(Ovalid.values)
Dvalid = pd.read_csv('./predictDCNN_valid.csv', header=None)
Dvalid = np.array(Dvalid.values)
whole_mat = []

for m in range(2):
    # morning peak
    OvalidAM = Ovalid[0 + T * m:AM_num + T * m]
    DvalidAM = Dvalid[0 + T * m:AM_num + T * m]
    depth = POI_mat.shape[1] * 2 + 2
    for k in range(AM_num):
        MAT = np.zeros([depth, 285, 285])
        for i in range(POI_mat.shape[1]):
            for j in range(285):
                MAT[0, :, j] = OvalidAM[k, :]
                MAT[1, j, :] = DvalidAM[k, :]
                MAT[i + 2, :, j] = POI_mat[:, i]
                MAT[POI_mat.shape[1] + 2 + i, j, :] = POI_mat[:, i]
        MAT[MAT < 0] = 0
        lst = [0.1937, 0.1906, 0.0641, -0.0076, -0.0148, -0.0216, -0.0084, -0.0112, -0.045, 0.0878, -0.0497, 0.0237,
               -0.0235, 0, -0.0610, -0.0673]
        for i in range(16):
            if (i == 0):
                OD_mat = pow(MAT[i] + 1, lst[i])
            else:
                OD_mat = OD_mat * pow(MAT[i] + 1, lst[i])
        OD_mat = OD_mat * pow(np.e, 0.8110) - 1
        OD_mat = OD_mat / Max_od
        OD_mat = OD_mat[:, :, np.newaxis]
        whole_mat.append(OD_mat)

    # off peak
    OvalidAM = Ovalid[11 - num_stamp + T * m:35 - num_stamp + T * m]
    DvalidAM = Dvalid[11 - num_stamp + T * m:35 - num_stamp + T * m]
    depth = POI_mat.shape[1] * 2 + 2
    for k in range(24):
        MAT = np.zeros([depth, 285, 285])
        for i in range(POI_mat.shape[1]):
            for j in range(285):
                MAT[0, :, j] = OvalidAM[k, :]
                MAT[1, j, :] = DvalidAM[k, :]
                MAT[i + 2, :, j] = POI_mat[:, i]
                MAT[POI_mat.shape[1] + 2 + i, j, :] = POI_mat[:, i]
        MAT[MAT < 0] = 0
        lst = [0.0967, 0.1285, 0.0288, -0.0158, 0.0197, -0.0207, 0, -0.0386, -0.0320, 0.0188, -0.0358, 0.0152, -0.0042,
               0.0110, -0.0270, -0.0059]
        for i in range(16):
            if (i == 0):
                OD_mat = pow(MAT[i] + 1, lst[i])
            else:
                OD_mat = OD_mat * pow(MAT[i] + 1, lst[i])
        OD_mat = OD_mat * pow(np.e, 1.7581) - 1
        OD_mat = OD_mat / Max_od
        OD_mat = OD_mat[:, :, np.newaxis]
        whole_mat.append(OD_mat)

    # evening peak
    OvalidAM = Ovalid[35 - num_stamp + T * m:42 - num_stamp + T * m]
    DvalidAM = Dvalid[35 - num_stamp + T * m:42 - num_stamp + T * m]
    depth = POI_mat.shape[1] * 2 + 2
    for k in range(7):
        MAT = np.zeros([depth, 285, 285])
        for i in range(POI_mat.shape[1]):
            for j in range(285):
                MAT[0, :, j] = OvalidAM[k, :]
                MAT[1, j, :] = DvalidAM[k, :]
                MAT[i + 2, :, j] = POI_mat[:, i]
                MAT[POI_mat.shape[1] + 2 + i, j, :] = POI_mat[:, i]
        MAT[MAT < 0] = 0
        lst = [0.1771, 0.1822, 0.0905, -0.0789, 0.0395, -0.0309, 0.0328, -0.0727, -0.0717, 0.0552, -0.0078, 0, -0.0235,
               0, -0.0182, -0.0516]
        for i in range(16):
            if (i == 0):
                OD_mat = pow(MAT[i] + 1, lst[i])
            else:
                OD_mat = OD_mat * pow(MAT[i] + 1, lst[i])
        OD_mat = OD_mat * pow(np.e, 1.1103) - 1
        OD_mat = OD_mat / Max_od
        OD_mat = OD_mat[:, :, np.newaxis]
        whole_mat.append(OD_mat)

OD_valid = np.array(whole_mat)
np.save('OD_valid.npy', OD_valid)

Otest = pd.read_csv('./predictOCNN_test.csv', header=None)
Otest = np.array(Otest.values)
Dtest = pd.read_csv('./predictDCNN_test.csv', header=None)
Dtest = np.array(Dtest.values)
whole_mat = []

for m in range(2):
    # morning peak
    OtestAM = Otest[0 + T * m:AM_num + T * m]
    DtestAM = Dtest[0 + T * m:AM_num + T * m]
    depth = POI_mat.shape[1] * 2 + 2
    for k in range(AM_num):
        MAT = np.zeros([depth, 285, 285])
        for i in range(POI_mat.shape[1]):
            for j in range(285):
                MAT[0, :, j] = OtestAM[k, :]
                MAT[1, j, :] = DtestAM[k, :]
                MAT[i + 2, :, j] = POI_mat[:, i]
                MAT[POI_mat.shape[1] + 2 + i, j, :] = POI_mat[:, i]
        MAT[MAT < 0] = 0
        lst = [0.1937, 0.1906, 0.0641, -0.0076, -0.0148, -0.0216, -0.0084, -0.0112, -0.045, 0.0878, -0.0497, 0.0237,
               -0.0235, 0, -0.0610, -0.0673]
        for i in range(16):
            if (i == 0):
                OD_mat = pow(MAT[i] + 1, lst[i])
            else:
                OD_mat = OD_mat * pow(MAT[i] + 1, lst[i])
        OD_mat = OD_mat * pow(np.e, 0.8110) - 1
        OD_mat = OD_mat / Max_od
        OD_mat = OD_mat[:, :, np.newaxis]
        whole_mat.append(OD_mat)

    # off peak
    OtestAM = Otest[11 - num_stamp + T * m:35 - num_stamp + T * m]
    DtestAM = Dtest[11 - num_stamp + T * m:35 - num_stamp + T * m]
    depth = POI_mat.shape[1] * 2 + 2
    for k in range(24):
        MAT = np.zeros([depth, 285, 285])
        for i in range(POI_mat.shape[1]):
            for j in range(285):
                MAT[0, :, j] = OtestAM[k, :]
                MAT[1, j, :] = DtestAM[k, :]
                MAT[i + 2, :, j] = POI_mat[:, i]
                MAT[POI_mat.shape[1] + 2 + i, j, :] = POI_mat[:, i]
        MAT[MAT < 0] = 0
        lst = [0.0967, 0.1285, 0.0288, -0.0158, 0.0197, -0.0207, 0, -0.0386, -0.0320, 0.0188, -0.0358, 0.0152, -0.0042,
               0.0110, -0.0270, -0.0059]
        for i in range(16):
            if (i == 0):
                OD_mat = pow(MAT[i] + 1, lst[i])
            else:
                OD_mat = OD_mat * pow(MAT[i] + 1, lst[i])
        OD_mat = OD_mat * pow(np.e, 1.7581) - 1
        OD_mat = OD_mat / Max_od
        OD_mat = OD_mat[:, :, np.newaxis]
        whole_mat.append(OD_mat)

    # evening peak
    OtestAM = Otest[35 - num_stamp + T * m:42 - num_stamp + T * m]
    DtestAM = Dtest[35 - num_stamp + T * m:42 - num_stamp + T * m]
    depth = POI_mat.shape[1] * 2 + 2
    for k in range(7):
        MAT = np.zeros([depth, 285, 285])
        for i in range(POI_mat.shape[1]):
            for j in range(285):
                MAT[0, :, j] = OtestAM[k, :]
                MAT[1, j, :] = DtestAM[k, :]
                MAT[i + 2, :, j] = POI_mat[:, i]
                MAT[POI_mat.shape[1] + 2 + i, j, :] = POI_mat[:, i]
        MAT[MAT < 0] = 0
        lst = [0.1771, 0.1822, 0.0905, -0.0789, 0.0395, -0.0309, 0.0328, -0.0727, -0.0717, 0.0552, -0.0078, 0, -0.0235,
               0, -0.0182, -0.0516]
        for i in range(16):
            if (i == 0):
                OD_mat = pow(MAT[i] + 1, lst[i])
            else:
                OD_mat = OD_mat * pow(MAT[i] + 1, lst[i])
        OD_mat = OD_mat * pow(np.e, 1.1103) - 1
        OD_mat = OD_mat / Max_od
        OD_mat = OD_mat[:, :, np.newaxis]
        whole_mat.append(OD_mat)

OD_test = np.array(whole_mat)
np.save('OD_test.npy', OD_test)
