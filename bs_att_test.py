import torch

import numpy as np
import matplotlib.pyplot as plt


person = "S5"
device = "cpu"
th_value = 0.25


pth_path_dict = {
    "S1":"weight_att\S1_time=20220528-192136_device=cuda_epoch=40_accuracy=85.6863.pth",
    "S2":"weight_att\S2_time=20220528-200010_device=cuda_epoch=40_accuracy=84.2157.pth",
    "S3":"weight_att\S3_time=20220528-200426_device=cuda_epoch=40_accuracy=85.1961.pth",
    "S4":"weight_att\S4_time=20220528-201531_device=cuda_epoch=30_accuracy=86.0784.pth",
    "S5":"weight_att\S5_time=20220528-201903_device=cuda_epoch=30_accuracy=90.2941.pth",
}



pth_path = pth_path_dict[person]

static_dict = torch.load(pth_path, map_location=device)
att_test = static_dict["att.weight"]
att_test_np = torch.squeeze(att_test).detach().numpy()
att_test_np_sum = np.sum(att_test_np,axis=0)

max_x_value = len(att_test_np_sum)+1

plt.bar(range(1,max_x_value),att_test_np_sum/max(att_test_np_sum))

plt.title(person)

ax = plt.gca()

ax.xaxis.set_major_locator(plt.MultipleLocator(1))
plt.xlim(0,max_x_value)
plt.hlines(0,0,max_x_value,colors="black")
plt.hlines( th_value,0,max_x_value,colors="r")
plt.hlines(-th_value,0,max_x_value,colors="r")

plt.show()