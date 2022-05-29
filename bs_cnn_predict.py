from tkinter.messagebox import NO
import torch
import numpy as np
import glob

from P300Data import rc_gt_dict,rc_char_dict
from bs_nets import CNN_1D

import os
import time


start_time = time.time()
time_str = time.strftime("%Y%m%d-%H%M%S")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pth_path_dict = {
    "S1":"weight_save/S1_time=20220527-163647_device=cuda_epoch=15_accuracy=97.0833.pth",
    "S2":"weight_save\S2_time=20220528-154425_device=cuda_epoch=20_accuracy=93.0392.pth",
    "S3":"weight_save\S3_time=20220528-163607_device=cuda_epoch=30_accuracy=93.9216.pth",
    "S4":"weight_save\S4_time=20220528-165418_device=cuda_epoch=15_accuracy=87.8431.pth",
    "S5":"weight_save\S5_time=20220528-170447_device=cuda_epoch=20_accuracy=92.9412.pth",
}

person = "S5"
pth_path = pth_path_dict[person]
# char = "char17"


def load_test_data(data_path):
    data_np = np.loadtxt(data_path,delimiter=",")
    data = torch.from_numpy(data_np.T).float()
    return data.unsqueeze(0)


# print(csv_pathL)
# csv_path = csv_pathL[0]




if __name__ == "__main__":
    model = CNN_1D()
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model = model.to(device)
    model.eval()

    # "MF52I"
    charL = ["char13","char14","char15","char16","char17","char18","char19","char20","char21","char22"]

    # for char,gt_char in zip(charL,gt_charL):
    for char in charL:
        round_num = None
        last_round_num = -1
        result_dict = {}

        csv_pathL = glob.glob(f"data/particle_data/test/{person}/*char={char}*.csv")
        # print(f"{rc_char_dict[gt_char]} {rc_gt_dict[gt_char]}")
        for csv_path in csv_pathL:
            # print(csv_path)
            inputs = load_test_data(csv_path)
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs, 1)

            propL = os.path.splitext(os.path.basename(csv_path))[0].split("_")
            round_text = propL[3].split("=")[-1]
            round_num = int(round_text.split("-")[0])
            rc_text = propL[4].split("=")[-1]


            if predicted.item():
                if not rc_text in result_dict:
                    result_dict[rc_text] = 0
                result_dict[rc_text] += 1

            # if last_round_num != round_num and round_num != 0:
            #     print(f"round={round_num}",sorted(result_dict.items(),key=lambda x:x[1],reverse=True))
            #     last_round_num = round_num
        sorted_dict = sorted(result_dict.items(),key=lambda x:x[1],reverse=True)
        # print(f"char={char}",sorted_dict)
        ansL = [-1]*2

        for k,v in sorted_dict:
            k = int(k)
            if k in range(1,7) and ansL[0] == -1:
                ansL[0] = k
            if k in range(7,13) and ansL[1] == -1:
                ansL[1] = k
            if -1 not in ansL:
                break
        ans_k = None
        for k,v in rc_gt_dict.items():
            if v == ansL:
                ans_k = k
                break

        # print(f"char={char}",rc_char_dict[k])
        if ans_k is not None:
            print(rc_char_dict[k],end=" ")
        else:
            print("N/A",end=" ")