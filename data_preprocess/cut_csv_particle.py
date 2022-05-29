import os
import glob
import numpy as np

event_data_pathL = glob.glob("../data/event_data/*/*.csv")
tt_data_pathL = glob.glob("../data/filter_csv/*/*.csv")
out_dir = "../data/particle_data"

def cut_train_data():
    for event_data_path,tt_data_path in zip(event_data_pathL,tt_data_pathL):
        event_type = os.path.basename(os.path.dirname(event_data_path)).split("_")
        tt_type = os.path.basename(os.path.dirname(tt_data_path)).split("_")

        # assert tt_type[:2] == event_type[:2], f"tt_type={tt_type} not match event_type={event_type}"
        people,tt = tt_type[:2]
        if tt != "train":
            continue

        if not os.path.exists(f"{out_dir}/{tt}/{people}"):
            os.mkdir(f"{out_dir}/{tt}/{people}")

        event_data = np.loadtxt(event_data_path,np.int,delimiter=",")
        data_all = np.loadtxt(tt_data_path,delimiter=",")

        data_name_temple = "{}_{}_char={}_round={}-{}_rc={}.csv"

        cnt_char,_ = event_data[0]
        round_i,round_j = 0,0
        for i in range(1,event_data.shape[0]-1):
            cnt_rc,cnt_start = event_data[i]
            if cnt_rc == 100:
                round_i += 1 
                round_j = 0
                continue
            data_name = data_name_temple.format(tt,people,cnt_char,round_i,round_j,cnt_rc)
            np.savetxt(f"{out_dir}/{tt}/{people}/{data_name}",data_all[cnt_start:cnt_start+160,:],fmt="%.12f",delimiter=",")
            round_j += 1

def cut_test_data():
    for event_data_path,tt_data_path in zip(event_data_pathL,tt_data_pathL):
        event_type = os.path.basename(os.path.dirname(event_data_path)).split("_")
        tt_type = os.path.basename(os.path.dirname(tt_data_path)).split("_")
        cnt_char = os.path.splitext(os.path.basename(event_data_path))[0].split("_")[-1]

        # assert tt_type[:2] == event_type[:2], f"tt_type={tt_type} not match event_type={event_type}"
        people,tt = tt_type[:2]
        if tt != "test":
            continue

        if not os.path.exists(f"{out_dir}/{tt}/{people}"):
            os.mkdir(f"{out_dir}/{tt}/{people}")

        event_data = np.loadtxt(event_data_path,np.int,delimiter=",")
        data_all = np.loadtxt(tt_data_path,delimiter=",")

        data_name_temple = "{}_{}_char={}_round={}-{}_rc={}.csv"


        round_i,round_j = 0,0
        for i in range(1,event_data.shape[0]-1):
            cnt_rc,cnt_start = event_data[i]
            if cnt_rc == 100:
                round_i += 1 
                round_j = 0
                continue
            data_name = data_name_temple.format(tt,people,cnt_char,round_i,round_j,cnt_rc)
            np.savetxt(f"{out_dir}/{tt}/{people}/{data_name}",data_all[cnt_start:cnt_start+160,:],fmt="%.12f",delimiter=",")
            round_j += 1



if __name__ == '__main__':
    cut_test_data()
