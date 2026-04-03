import numpy as np
import os
from bisect import bisect_right
import time
import itertools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

base_dir = Path(__file__).resolve().parent

Reso = 8  #分辨率
RANGE = 'WCT600' # data file name
# RANGE2 = '5501_6000'
# left = 2500
# right = 3000

def sort_by_number(item):
    # 提取文件名中的数字部分
    number = item.split('_')[1]
    return int(number)

filepath =  base_dir / "MD_sim_examples" / f"{RANGE}"
Nfile = len(os.listdir(filepath))
Flist = sorted(os.listdir(filepath))


# Initialize input and output arrays
in_train = np.zeros((Nfile, 12, Reso, Reso, Reso))
out_train = np.zeros((Nfile, 6, Reso, Reso, Reso))

for r in range(Nfile):
    start_time = time.time()  # 记录开始时间
    # file_list = sorted(os.listdir(path), key=sort_by_number)

    with open(filepath+f'/{Flist[r]}', 'r+') as file:
        total_lines = file.readlines()[5:]
        spacial_range = total_lines[:3]
        lines = total_lines[4:]
        atom_labels = [int(line.split()[1]) for line in lines]   # labels of all atoms in the file
        vir_stress = np.array([line.split()[8:14] for line in lines], dtype=float)
        WC = np.array([line.split()[-9:] for line in lines], dtype=float)
        # print(WC.shape)
        t1 = np.array([line.split()[2:5] for line in lines], dtype=float)

    l = [list(map(float, x.split())) for x in spacial_range]

    # Generate intervals
    intervals = np.zeros((3,Reso))   # Separating values of intervals
    for i in range(3):
        h = (l[i][1]-l[i][0])/Reso
        for j in range(Reso):
            intervals[i,j] = j*h
    # Generate interval labels
    intervals_values = list(itertools.product(range(Reso), range(Reso), range(Reso)))
    # Determine atom labels

    info = []
    for i in range(len(lines)):
        a = bisect_right(intervals[0], t1[i, 0]) - 1
        b = bisect_right(intervals[1], t1[i, 1]) - 1
        c = bisect_right(intervals[2], t1[i, 2]) - 1
        info.append(intervals_values.index((a, b, c)))
    Info = np.array(info)

    # Process each block
    for i in itertools.product(range(Reso), range(Reso), range(Reso)):
        label_values = intervals_values.index((i[0], i[1], i[2]))
        intervals_labels = np.where(Info == label_values)[0].tolist()
        # print('length of intervals_labels=',max(intervals_labels))
        new1 = [atom_labels[m] for m in intervals_labels] #挑选出该block所属的原子
        for j in range(3):
            in_train[r][j][i[0]][i[1]][i[2]] = new1.count(j+1)/len(new1)  # 每个block三种原子的浓度 （平均值）
        new3 = [WC[m] for m in intervals_labels]   # WC parameter       
        new3 = np.array(new3)     
        for j in range(3,12):
            in_train[r][j][i[0]][i[1]][i[2]] = np.mean(new3[:,j-3])

        new2 = [vir_stress[m] for m in intervals_labels]   # viral stress
        new2 = np.array(new2)
        for j in range(6):
            out_train[r][j][i[0]][i[1]][i[2]] = np.mean(new2[:, j])  # 每个block xx yy zz virtual stress 平均值

    end_time = time.time()  # 记录结束时间
    run_time = end_time - start_time  # 计算执行时间
    print(f"循环 {Flist[r]} 的执行时间: {run_time}秒")


# Save input and output arrays
save_path = base_dir / "block_averaged_data"
# np.save(file=save_path + f"/INPUT_train{RANGE}_X.npy", arr=in_train)
# np.save(file=save_path + f"/OUTPUT_stress1118_total{RANGE}.npy", arr=out_train)