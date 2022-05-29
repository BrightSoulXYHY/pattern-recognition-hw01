import numpy as np
from scipy import signal

import os
import glob


csv_pathL = glob.glob("data/extract_csv/*/*.csv")
out_dir = "data/filter_csv"




wsamp = 250
Wn=[2*i/wsamp for i in [1,20]]
b,a = signal.butter(4, Wn, 'bandpass')


for csv_path in csv_pathL:
# csv_path = csv_pathL[0]

    csv_dirname = os.path.basename(os.path.dirname(csv_path))
    csv_basename = os.path.basename(csv_path)


    if not os.path.exists(f"{out_dir}/{csv_dirname}"):
        os.mkdir(f"{out_dir}/{csv_dirname}")


    data_all = np.loadtxt(csv_path,delimiter=",")
    data_out = np.zeros(data_all.shape)

    for i in range(data_all.shape[-1]):
        filtedData = signal.filtfilt(b,a,data_all[:,i])
        data_out[:,i] = filtedData

    np.savetxt(f"{out_dir}/{csv_dirname}/{csv_basename}",data_out,fmt="%.12f",delimiter=",")