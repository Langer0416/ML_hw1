import torch
from torchvision import transforms
import model
from dataset import Predict_dataset
import matplotlib.pyplot as plt

if __name__ == '__main__':

    weight_path = "./weight/Resnet50.pth"

    labels = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
    predict_label = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device is : {}".format(device))

    Resnet50_model = model.CustomResNet50(12)

    Resnet50_model.load_state_dict(torch.load(weight_path))

    Resnet50_model.eval()

    Resnet50_model = Resnet50_model.to(device)

    transforms = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor()])

    test_data = Predict_dataset(img_path="./test", transform=transforms)

    num_image = 5
    fig, axs = plt.subplots(1, num_image, figsize=(15, 3))

    with torch.no_grad():
        for data in test_data:
            data = data.to(device)
            outputs = Resnet50_model(data)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy().astype(int)
            predict_label.append(labels[predicted[0]])

    for i in range(5):
        axs[i].imshow(test_data[i].squeeze(0).permute(1, 2, 0))
        axs[i].axis('off')
        axs[i].set_title(predict_label[i])
    plt.savefig('./PIC/predict.png')
    plt.show()
