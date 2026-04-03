# training data size = 4000
import itertools
import random
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler


base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "block_averaged_data"

data_index = ["650", "700", "750", "800", "850", "900", "950", "1000", "1100"]

x1 = np.load(data_dir / "INPUT_train1-500WCrandom.npy")  # shape is (100, 16, 16, 16, 3)
x2 = np.load(data_dir / "INPUT_trainWCT600_X.npy")
y1 = np.load(data_dir / "OUTPUT_stress1118_totalWCrandom.npy")  # shape is (100, 16, 16, 16, 3)
y2 = np.load(data_dir / "OUTPUT_stress1118_totalWCT600.npy")
x_atrain = np.concatenate((x1, x2), axis=0)
y_atrain = np.concatenate((y1, y2), axis=0)

for i in data_index:
    x = np.load(data_dir / f"INPUT_trainWCT{i}_X.npy")
    y = np.load(data_dir / f"OUTPUT_stress1118_totalWCT{i}.npy")
    x_atrain = np.concatenate((x_atrain, x), axis=0)  # size 2000*3*16*16*16
    y_atrain = np.concatenate((y_atrain, y), axis=0)  # size 2000*3*16*16*16

N_cell = len(x_atrain)
Reso = 4

x_new = np.zeros((N_cell, 12, Reso, Reso, Reso))
y_new = np.zeros((N_cell, 3, Reso, Reso, Reso))
for i in itertools.product(range(N_cell), range(Reso), range(Reso), range(Reso)):
    r = i[1] * int(8 / Reso)
    s = i[2] * int(8 / Reso)
    t = i[3] * int(8 / Reso)
    for j in range(12):
        x_new[i[0]][j][i[1]][i[2]][i[3]] = np.sum(
            x_atrain[i[0], j, r : r + int(8 / Reso), s : s + int(8 / Reso), t : t + int(8 / Reso)]
        ) / (int(8 / Reso) ** 3)
    for k in range(6):
        y_new[i[0]][k][i[1]][i[2]][i[3]] = np.sum(
            y_atrain[i[0], k, r : r + int(8 / Reso), s : s + int(8 / Reso), t : t + int(8 / Reso)]
        ) / (int(8 / Reso) ** 3)

x_new_concen = x_new[:, :3, :, :, :]
x_new_wc = x_new[:, 3:, :, :, :]

x_concen = x_new_concen.reshape([N_cell, 3 * Reso * Reso * Reso])
x_wc = x_new_wc.reshape([N_cell, 9 * Reso * Reso * Reso])
y = y_new.reshape([N_cell, 3 * Reso * Reso * Reso])

my_scaler_concen = MinMaxScaler()
x_all_concen = my_scaler_concen.fit_transform(x_concen)

my_scaler_wc = MinMaxScaler()
x_all_wc = my_scaler_wc.fit_transform(x_wc)

my_scaler = MinMaxScaler()
y_all = my_scaler.fit_transform(y)

X_all_concen = x_all_concen.reshape([N_cell, 3, Reso, Reso, Reso])
X_all_wc = x_all_wc.reshape([N_cell, 9, Reso, Reso, Reso])
X_all = np.concatenate((X_all_concen, X_all_wc), axis=1)

Y_all = y_all.reshape([N_cell, 3, Reso, Reso, Reso])

index = [i for i in range(N_cell)]
random.shuffle(index)
X_random = X_all[index]
Y_random = Y_all[index]

# np.save(base_dir / "X_WC_3500.npy", X_random)
# np.save(base_dir / "Y_WC_3500.npy", Y_random)
