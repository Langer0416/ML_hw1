import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import train, model
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

if __name__ == '__main__':

    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    parser = argparse.ArgumentParser(description='PyTorch Brain_Tumor Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--e', default=5, type=int, help='Your epoch')
    args = parser.parse_args()

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

    train_dataset = ImageFolder(root='./train', transform=transform)

    datasets = train_val_dataset(train_dataset)
    print(len(datasets['train']))
    print(len(datasets['val']))

    train_loader = DataLoader(datasets['train'], batch_size=32, shuffle=True, num_workers=0)
    valid_loader = DataLoader(datasets['val'], batch_size=32, shuffle=True, num_workers=0)


    print(train_dataset.class_to_idx)
    # print(train_dataset.classes) #emotion name

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device is :{}".format(device))

    train_net = model.CustomResNet50(num_classes=12).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(train_net.parameters(), lr = args.lr,
                          momentum=0.9, weight_decay=5e-4)

    for epoch in range(1, args.e+1):
        get_item = train.train(epoch, train_net, device, train_loader, valid_loader, optimizer, criterion, args.e)

        train_losses.append([epoch, get_item[0]])
        valid_losses.append([epoch, get_item[1]])
        train_accuracies.append([epoch, get_item[2]])
        valid_accuracies.append([epoch, get_item[3]])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()