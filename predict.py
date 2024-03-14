import os

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import model
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor()])

    weight_path = "./weight/Resnet50.pth"

    print(os.getcwd())

    test_dataset = ImageFolder(root=os.path.join(os.getcwd(), 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)

    classes = test_dataset.classes

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device is : {}".format(device))

    Resnet50_model = model.CustomResNet50(len(classes))

    Resnet50_model.load_state_dict(torch.load(weight_path))

    Resnet50_model.eval()

    Resnet50_model = Resnet50_model.to(device)

    correct = 0
    total = 0

    y_pred = []
    y_true = []

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    with torch.no_grad():
        correct1 = 0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = Resnet50_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels)
            for i in range(1):
                label = labels[i]
                y_true.append(labels[i].cpu().numpy())
                y_pred.append(predicted[i].cpu().numpy())
                # print("label is : {}, ".format(label))
                class_correct[label] += c[i]
                class_total[label] += 1

    print(f"Accuracy on Test Set: {100 * correct / total}%")
    print("Correct is : {}, Total is : {}".format(correct, total))

    for i in range(len(classes)):
        print("Class : {} is {} and Total number is {}".format(i, classes[i], class_total[i]))

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    cf_matrix = confusion_matrix(y_true, y_pred)
    per_cls_acc = cf_matrix.diagonal() / cf_matrix.sum(axis=1)

    print(classes)
    print(per_cls_acc)  # 顯示每個class的Accuracy
    print("Plot confusion matrix")

    print("Cf matrix")
    print(cf_matrix)

    df_cm = pd.DataFrame(cf_matrix, classes, classes)  # https://sofiadutta.github.io/datascience-ipynbs/pytorch/Image-Classification-using-PyTorch.html
    plt.figure(figsize=(14, 10))
    sns.heatmap(df_cm, cmap='BuGn', annot=True, fmt='d')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.savefig("confusion_matrix.png")
    plt.show()