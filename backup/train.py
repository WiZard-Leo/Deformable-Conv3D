import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

import pdb

from torch.cuda.amp import autocast, GradScaler

from data.dataset import CTDataset, CTSubDataset
from net import Net

scaler = GradScaler()

train_dataset = CTDataset(mode='train')
test_dataset = CTDataset(mode='test')

# train_dataset = CTSubDataset(train_dataset, ratio=0.01)
# test_dataset = CTSubDataset(test_dataset, ratio=0.01)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=8, pin_memory=True, prefetch_factor=8)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=8, pin_memory=True, prefetch_factor=8)

device = 'cuda:4'
model = Net().to(device)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer = optim.SGD(model.parameters(), lr=0.01)

criterion = nn.BCEWithLogitsLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):  
        bz = len(target)
        label = torch.zeros((bz,5))
        for _ in range(bz):
            idxes = target[_][...,-1].reshape(-1)
            for idx in idxes:
                label[_][idx] = 1
        data = [d.to(device) for d in data]
        data = torch.stack(data)
        target = label.to(device)
        # data, target = data.to(device), label.to(device)

        optimizer.zero_grad()
        with autocast():
            output = model(data)
            # loss = F.nll_loss(output, target)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
#
# A simple test procedure to measure the STN performances on MNIST.
#
def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        total_samples = 0
        for data, target in test_loader:
            # data, target = data.to(device), target.to(device)
            bz = len(target)
            label = torch.zeros((bz,5))
            for _ in range(bz):
                idxes = target[_][...,-1].reshape(-1)
                for idx in idxes:
                    label[_][idx] = 1
            data = [d.to(device) for d in data]
            data = torch.stack(data)
            target = label.to(device)
            with autocast():
                output = model(data)

                # sum up batch loss
                # test_loss += F.nll_loss(output, target, size_average=False).item()

                test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            
            # pred = output.max(1, keepdim=True)[1]
            # correct += pred.eq(target.view_as(pred)).sum().item()

            # 对输出应用sigmoid以获得概率
            pred = torch.sigmoid(output)
            pred = (pred > 0.5).float()
            # correct += (pred == target).float().sum().item()
            correct += (pred * target).sum().item()
            # total_samples += target.numel()
            total_samples += target.sum().item()


        # test_loss /= len(test_loader.dataset)
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
        #       .format(test_loss, correct, len(test_loader.dataset),
        #               100. * correct / len(test_loader.dataset)))

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
            .format(test_loss, correct, total_samples,
                    100. * correct / total_samples))
        
        
def save_checkpoint(epoch, model, optimizer, scaler, checkpoint_path='checkpoint.pth'):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }
    torch.save(state, checkpoint_path)

if __name__ == '__main__':
    epochs = 20
    for epoch in range(epochs):
        train(epoch=epoch)
        test()
        if epoch % 2 == 0:
            save_checkpoint(epoch, model, optimizer, scaler, checkpoint_path=f'{epoch}.pth')
