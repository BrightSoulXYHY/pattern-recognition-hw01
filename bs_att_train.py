import torch
from torch.utils.data import DataLoader

from bs_nets import CNN_1D,Attention_CNN
from P300Data import P300TrainData,P300ValidData

import time
from tqdm import tqdm

# 超参数
learning_rate = 0.001
num_epochs = 30
continue_train = True
pth_path_dict = {
    "S1":"weight_save/S1_time=20220527-163647_device=cuda_epoch=15_accuracy=97.0833.pth",
    "S2":"weight_save\S2_time=20220528-154425_device=cuda_epoch=20_accuracy=93.0392.pth",
    "S3":"weight_save\S3_time=20220528-163607_device=cuda_epoch=30_accuracy=93.9216.pth",
    "S4":"weight_save\S4_time=20220528-165418_device=cuda_epoch=15_accuracy=87.8431.pth",
    "S5":"weight_save\S5_time=20220528-170447_device=cuda_epoch=20_accuracy=92.9412.pth",
}

person = "S5"
pth_path = pth_path_dict[person]


start_time = time.time()
time_str = time.strftime("%Y%m%d-%H%M%S")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# S1_train_dataset = P300TrainData(personL=["S1","S2","S3","S4","S5"])
S1_train_dataset = P300TrainData(personL=[person])
S1_test_dataset  = P300TrainData(personL=[person],train=False)
train_loader = DataLoader(S1_train_dataset, batch_size=120, shuffle=True, num_workers=2)
test_loader = DataLoader(S1_test_dataset, batch_size=40, shuffle=False, num_workers=2)

S1_valid_dataset = P300ValidData(person=person)
valid_loader = DataLoader(S1_valid_dataset, batch_size=60, shuffle=False, num_workers=2)



save_text = f'weight_att/{person}_time={time_str}_device={device}'+'_epoch={:02d}_accuracy={:.4f}.pth'

if __name__ == '__main__':
    
    cnn = CNN_1D()
    
    if continue_train:
        cnn.load_state_dict(torch.load(pth_path, map_location=device))
        print(f"[{time.time()-start_time:.2f}] Load state_dict done")
    cnn = cnn.to(device)
    cnn.eval()

    model = Attention_CNN()
    model = model.to(device)
    # 设置损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        val_loss = 0
        
        test_correct = 0
        test_sz_total = 0
        if epoch > 20:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate*0.1)

        
        model.train()
        pbar = tqdm(train_loader)
        for iteration,[inputs, targets] in enumerate(pbar):
            inputs,targets = inputs.to(device),targets.to(device)
            optimizer.zero_grad()
            
            att_out = model(inputs)
            outputs = cnn(att_out)

            loss = criterion(outputs, targets)

            loss.backward()
            # 更新参数
            optimizer.step()

            total_loss += loss.item()
            val_loss = total_loss / (iteration + 1)
            

            desc_str = f"{'Train':8s} [{epoch + 1}/{num_epochs}] loss:{val_loss:.6f}"
            pbar.desc = f"{desc_str:40s}"
        
        model.eval()
        pbar = tqdm(test_loader)
        # pbar = tqdm(train_loader)
        for iteration,[inputs, targets] in enumerate(pbar):
            inputs,targets = inputs.to(device),targets.to(device)
            
            att_out = model(inputs)
            outputs = cnn(att_out)
            
            _, predicted = torch.max(outputs, 1)
            test_sz_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()
            accuracy = 100 * test_correct / test_sz_total
            desc_str = f"{'Test':8s} [{epoch + 1}/{num_epochs}] accuracy:{accuracy:.6f}"
            pbar.desc = f"{desc_str:40s}"

        pbar = tqdm(valid_loader)
        for iteration,[inputs, targets] in enumerate(pbar):
            inputs,targets = inputs.to(device),targets.to(device)
            
            att_out = model(inputs)
            outputs = cnn(att_out)

            _, predicted = torch.max(outputs, 1)
            test_sz_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()
            accuracy = 100 * test_correct / test_sz_total
            desc_str = f"{'Valid':8s} [{epoch + 1}/{num_epochs}] accuracy:{accuracy:.6f}"
            pbar.desc = f"{desc_str:40s}"
        
        if not (epoch + 1) % 5 or (not epoch):
            torch.save(model.state_dict(), save_text.format(epoch + 1,accuracy) )
