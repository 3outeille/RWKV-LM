import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.optim as optim
from tqdm import tqdm

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# Model
class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        seed_everything(42)
        
        self.N = 32 * 32
        self.linear1 = nn.Linear(in_features=self.N, out_features=self.N)
        self.linear2 = nn.Linear(in_features=self.N, out_features=self.N)
        self.linear3 = nn.Linear(in_features=self.N, out_features=self.N)
        self.linear4 = nn.Linear(in_features=self.N, out_features=num_classes)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)

        residual = x
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = F.relu(x) + residual
        x = self.linear3(x)
        x = F.relu(x) + residual
        x = self.linear4(x)        
        return  x
    
class MNISTloader:
    def __init__(
        self,
        batch_size: int = 100,
        data_dir: str = "./data/",
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        train_val_split: float = 0.1,
    ):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.train_val_split = train_val_split

        self.setup()

    def setup(self):
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.train_dataset = datasets.MNIST(
            self.data_dir, train=True, download=True, transform=transform
        )
        val_split = int(len(self.train_dataset) * self.train_val_split)
        train_split = len(self.train_dataset) - val_split

        self.train_dataset, self.val_dataset = random_split(
            self.train_dataset, [train_split, val_split]
        )
        self.test_dataset = datasets.MNIST(
            self.data_dir, train=False, download=True, transform=transform
        )

        print(
            "Image Shape:    {}".format(self.train_dataset[0][0].numpy().shape),
            end="\n\n",
        )
        print("Training Set:   {} samples".format(len(self.train_dataset)))
        print("Validation Set: {} samples".format(len(self.val_dataset)))
        print("Test Set:       {} samples".format(len(self.test_dataset)))

    def load(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

        return train_loader, val_loader, test_loader

def train(num_epochs, model, optimizer, criterion, train_loader, device):

    model.train().cuda() if (device.type == "cuda") else model.train().cpu()

    grad_acc = {}

    for epoch in range(num_epochs):
        
        grad_acc[f"epoch_{epoch}"] = []

        train_loss_running, train_acc_running = 0, 0

        progress_bar = tqdm(total=len(train_loader))

        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            _, predictions = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)

            loss.backward()
            
            # Average gradient per step
            with torch.no_grad():
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                avg_gradient = torch.norm(torch.cat([g.flatten() for g in gradients]), p=2) / len(gradients)
                grad_acc[f"epoch_{epoch}"].append(avg_gradient.cpu().item())

            # with torch.no_grad():
            #     for name, param in model.named_parameters():
            #         if name == "linear1.weight":
            #             grad_acc[f"epoch_{epoch}"].append(param.grad.norm(2).cpu().item())
            
            optimizer.step()

            train_loss_running += loss.item() * inputs.shape[0]
            train_acc_running += torch.sum(predictions == labels.data)

            progress_bar.update(1)

        progress_bar.close()

        train_loss = train_loss_running / len(train_loader.sampler)
        train_acc = train_acc_running / len(train_loader.sampler)
        
        # Average gradient per epoch
        grad_acc[f"epoch_{epoch}"] = np.mean(grad_acc[f"epoch_{epoch}"])

        info = "Epoch: {:3}/{} \t train_loss: {:.3f} \t train_acc: {:.3f}"
        print(info.format(epoch + 1, num_epochs, train_loss, train_acc))

    return grad_acc

if __name__ == "__main__":
    
    seed_everything(42)
    lr = 0.02
    num_epochs = 6
    batch_size = 100
    criterion = nn.CrossEntropyLoss()
    train_loader, _, _ = MNISTloader(train_val_split=0.95, batch_size=100).load()

    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    grad_acc = train(num_epochs, model, optimizer, criterion, train_loader, device)
    torch.save(model.state_dict(), "model.pt")
    # torch.save(grad_acc, "grad_acc_layer1.pt")
    torch.save(grad_acc, "grad_acc.pt")
