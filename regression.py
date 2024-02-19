import torch.nn as nn
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import sys

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        train_losses = []
        running_train_loss = 0.0
        model.train() 
        
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for ecg, vcg in train_progress_bar:
            ecg, vcg = ecg.float(), vcg.float()
            M = model(ecg)
            vcg_m = torch.transpose(torch.bmm(torch.transpose(ecg, 1,2), torch.transpose(M, 1,2)),1,2)
            
            loss = criterion(vcg_m, vcg)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        train_losses.append(loss.item())

        running_val_loss = 0.0
        val_losses = []
        model.eval()
        
        with torch.no_grad():
            for ecg, vcg in val_loader:
                ecg = ecg.float()
                vcg = vcg.float()
                vcg_m = model(ecg)
                vcg_m = torch.bmm(vcg_m, ecg)
                loss = criterion(vcg_m, vcg)
                
                running_val_loss += loss.item()
                val_losses.append(loss.item())

        print(f"Epoch {epoch+1}/{num_epochs}, \
                Train Loss: {running_train_loss/len(train_loader.dataset):.8f},\
                Val Loss: {running_val_loss/len(val_loader.dataset):.8f}")


class ECGDataset(Dataset):
    def __init__(self, dataset):
        self.data = [data[0].T[:12,:] for data in dataset]
        self.targets = [data[0].T[-3:,:] for data in dataset]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
      
class ECGregression(nn.Module):
    def __init__(self):
        super(ECGregression, self).__init__()
        
        # Define the 1D convolutional layers
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=36, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=36, out_channels=36, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=36, out_channels=36, kernel_size=3)
        
        self.fc1 = nn.Linear(36, 36)
        self.fc2 = nn.Linear(36, 36)
    
    def reconstruct(self, ecg):
        M = self.forward(ecg)
        vcg_m = torch.transpose(torch.bmm(torch.transpose(ecg, 1,2), torch.transpose(M, 1,2)),1,2)
        return vcg_m
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)  
        #print('----DEBUG MODEL----')
        #print(x.shape)
        x = x[:,:,-1]
        #print('----Flattening----')
        #print(x.shape)
        
        # Apply the fully connected layers
        x = self.fc1(x)
        x = nn.Tanh()(x)
        x = self.fc2(x)
        
        # Reshape the output to a 3x12 matrix
        x = x.view(-1, 3, 12)
        
        return x


