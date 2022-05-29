import os
import numpy as np

import torch
from torch.utils.data import Dataset


np.random.seed(1)

rc_gt_dict = {
    "101":[1,7], "102":[1,8], "103":[1,9], "104":[1,10], "105":[1,11], "106":[1,12],
    "107":[2,7], "108":[2,8], "109":[2,9], "110":[2,10], "111":[2,11], "112":[2,12],
    "113":[3,7], "114":[3,8], "115":[3,9], "116":[3,10], "117":[3,11], "118":[3,12],
    "119":[4,7], "120":[4,8], "121":[4,9], "122":[4,10], "123":[4,11], "124":[4,12],
    "125":[5,7], "126":[5,8], "127":[5,9], "128":[5,10], "129":[5,11], "130":[5,12],
    "131":[6,7], "132":[6,8], "133":[6,9], "134":[6,10], "135":[6,11], "136":[6,12],
}
rc_char_dict = {
    "101": "A", "102": "B", "103": "C", "104": "D", "105": "E", "106": "F",
    "107": "G", "108": "H", "109": "I", "110": "J", "111": "K", "112": "L",
    "113": "M", "114": "N", "115": "O", "116": "P", "117": "Q", "118": "R",
    "119": "S", "120": "T", "121": "U", "122": "V", "123": "W", "124": "X",
    "125": "Y", "126": "Z", "127": "1", "128": "2", "129": "3", "130": "4",
    "131": "5", "132": "6", "133": "7", "134": "8", "135": "9", "136": "0",
}

# for i in range(6):
#     for j in range(6):
#         print(f"\"{100+i*6+j+1}\": \"{chr(64+i*6+j+1)}\",")



# L = [i for i in range(10)]

# np.random.shuffle(L)

# print(L)



class P300TrainData(Dataset):
    def __init__(self, personL=["S1"],train=True):
    # def __init__(self, personL=["S1"]):
        self.train = train
        # self.transform = transform
        # self.target_transform = target_transform

        # self.pos_dataD = {}
        self.pos_dataL = []
        self.neg_dataL = []

        for person in personL:
            raw_folder = f"data/particle_data/train/{person}"
            L = os.listdir(raw_folder)
            for name in L:
                propL = os.path.splitext(name)[0].split("_")
                char = propL[2].split("=")[-1]
                rc = propL[4].split("=")[-1]
                if int(rc) in rc_gt_dict[char]:
                    self.pos_dataL.append(f"{raw_folder}/{name}")
                    # if rc not in self.pos_dataD:
                    #     self.pos_dataD[rc] = []
                    # self.pos_dataD[rc].append(f"{raw_folder}/{name}")

                else:
                    self.neg_dataL.append(f"{raw_folder}/{name}")

        # for k,v in self.pos_dataD.items():
        #     print(f"{k}={len(v)}")
        # print(len(self.pos_dataL),len(self.neg_dataL))


        extand_scale = len(self.neg_dataL)//len(self.pos_dataL)
        test_len = len(self.pos_dataL)//5
        train_data_pos = self.pos_dataL[test_len:]
        train_data_neg = self.neg_dataL[test_len*extand_scale:]
        

        np.random.shuffle(self.neg_dataL)
        # print(extand_scale)
        # print(len(train_data_pos),len(train_data_neg))
        # print((len(self.pos_dataL)-test_len)*extand_scale,len(self.neg_dataL)-test_len)

        # self.data_pathL = self.pos_dataL*extand_scale + self.neg_dataL
        # self.labelL = [1]*len(self.pos_dataL)*extand_scale + [0]*len(self.neg_dataL)
        if self.train:
            self.data_pathL = self.pos_dataL*extand_scale + self.neg_dataL
            self.labelL = [1]*len(self.pos_dataL)*extand_scale + [0]*len(self.neg_dataL)
            # self.data_pathL = train_data_pos*extand_scale + train_data_neg
            # self.labelL = [1]*len(train_data_pos)*extand_scale + [0]*len(train_data_neg)
        else:
            self.data_pathL = self.pos_dataL + self.neg_dataL
            self.labelL = [1]*len(self.pos_dataL) + [0]*len(self.neg_dataL)
            # self.data_pathL = self.pos_dataL[:test_len] + self.neg_dataL[:test_len*extand_scale]
            # self.labelL = [1]*test_len + [0]*test_len*extand_scale

        assert len(self.data_pathL) == len(self.labelL),"len(self.data_pathL) != len(self.labelL)"

    def __getitem__(self, index):
        data_path = self.data_pathL[index]
        data_np = np.loadtxt(data_path,delimiter=",")
        data = torch.from_numpy(data_np.T).float()
        label = self.labelL[index]
        return data, label

    def __len__(self) -> int:
        return len(self.data_pathL)


class P300ValidData(Dataset):
    def __init__(self, person="S1"):
        self.pos_dataL = []
        self.neg_dataL = []
        valid_char_dict = {
            "char13":"113",
            "char14":"106",
            "char15":"131",
            "char16":"128",
            "char17":"109",
        }
        # charL = ["char13","char14","char15","char16","char17"]
        # gt_charL = ["113","106","131","128","109"]

        raw_folder = f"data/particle_data/test/{person}"
        # csv_pathL = glob.glob(f"data/particle_data/test/{person}/*char={char}*.csv")
        L = os.listdir(raw_folder)
        for name in L:
            propL = os.path.splitext(name)[0].split("_")
            char = propL[2].split("=")[-1]
            rc = propL[4].split("=")[-1]
            if char not in valid_char_dict:
                continue
            



            if int(rc) in rc_gt_dict[valid_char_dict[char]]:
                self.pos_dataL.append(f"{raw_folder}/{name}")
            else:
                self.neg_dataL.append(f"{raw_folder}/{name}")


        self.data_pathL = self.pos_dataL + self.neg_dataL
        self.labelL = [1]*len(self.pos_dataL) + [0]*len(self.neg_dataL)

        assert len(self.data_pathL) == len(self.labelL),"len(self.data_pathL) != len(self.labelL)"

    def __getitem__(self, index):
        data_path = self.data_pathL[index]
        data_np = np.loadtxt(data_path,delimiter=",")
        data = torch.from_numpy(data_np.T).float()
        label = self.labelL[index]
        return data, label

    def __len__(self) -> int:
        return len(self.data_pathL)



if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # S1_train_dataset = P300TrainData(personL=["S1"])
    # S1_test_dataset  = P300TrainData(personL=["S1"],train=False)
    # print(len(S1_train_dataset),len(S1_test_dataset))
    S1_train_dataset = P300ValidData(person="S1")
    print(len(S1_train_dataset))


    # test_loader = DataLoader(dataset=S1_train_dataset, batch_size=100, shuffle=False)

    # for data, label in test_loader:
    #     # print(type(labels),)
    #     print(data.shape)
    #     break