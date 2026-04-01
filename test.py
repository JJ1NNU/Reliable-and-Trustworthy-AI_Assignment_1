import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

# CNN model
class CNN(nn.Module):
    def __init__(self, dataset='mnist'):
        super(CNN, self).__init__()
        if dataset == 'MNIST': # 1x28x28
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2), # 14x14
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)  # 7x7
            )
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 10)
            )
        elif dataset == 'CIFAR-10': # 3x32x32
            self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2), # 16x16
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 8x8
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2) # 4x4
            )
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 10)
            )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# CNN train function
def train_and_test(dataset, epochs=10, batch_size=128, lr=0.001):
    print(f"Training on {dataset.upper()} using {device}")

    # data load
    if dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset == "CIFAR-10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # setting
    model = CNN(dataset=dataset).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # # model save
    # save_path = f"{dataset}_cnn.pth"
    # torch.save(model.state_dict(), save_path)
    # print(f"saved model to: {save_path}")

    # test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    accuracy = 100 * correct / total

    print(f"Accuracy on {dataset.upper()}: {accuracy:.2f}%")

    return model

# Attack functions
def fgsm_targeted(dataset, model, x, target, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    # x는 normalize되지 않은 원본 데이터를 넣습니다(추후 visualization을 위해).
    # 따라서 정규화 필요 (normalize 함수 아래에 정의)
    norm_x = normalize(x_adv, dataset)
    output = model(norm_x)
    loss = nn.CrossEntropyLoss()(output, target)
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        x_adv = x_adv - eps * x_adv.grad.data.sign()
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv

def fgsm_untargeted(dataset, model, x, label, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    norm_x = normalize(x_adv, dataset)
    output = model(norm_x)
    loss = nn.CrossEntropyLoss()(output, label)
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        x_adv = x_adv + eps * x_adv.grad.data.sign()
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv

def pgd_targeted(dataset, model, x, target, k, eps, eps_step):
    x_adv = x.clone().detach().requires_grad_(True)
    # initialize: start from the clean input

    for _ in range(k):
        x_adv = x_adv.detach().requires_grad_(True)
        norm_x = normalize(x_adv, dataset)
        output = model(norm_x)
        loss = nn.CrossEntropyLoss()(output, target)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv - eps_step * x_adv.grad.data.sign()
            x_adv = torch.clamp(x_adv, x-eps, x+eps)
            x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv

def pgd_untargeted(dataset, model, x, label, k, eps, eps_step):
    x_adv = x.clone().detach().requires_grad_(True)

    for _ in range(k):
        x_adv = x_adv.detach().requires_grad_(True)
        norm_x = normalize(x_adv, dataset)
        output = model(norm_x)
        loss = nn.CrossEntropyLoss()(output, label)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + eps_step * x_adv.grad.data.sign()
            x_adv = torch.clamp(x_adv, x - eps, x + eps)
            x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv

def normalize(x, dataset):
    if dataset == 'MNIST':
        return transforms.Normalize((0.1307,), (0.3081,))(x)
    return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(x)

# Evaluate attack success rate
def evaluate_attack(dataset, model, eps_list, k, eps_step):
    model.eval()
    os.makedirs('results', exist_ok=True)

    # 정규화 안된 원본 이미지
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == 'MNIST':
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset == "CIFAR-10":
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    loader = DataLoader(test_set, batch_size=1, shuffle=False)

    attack_list= ['FGSM_Targeted', 'FGSM_Untargeted', 'PGD_Targeted', 'PGD_Untargeted']

    for eps in eps_list:
        for attack in attack_list:
            success = 0
            visualize_set = []

            for i, (img, label) in enumerate(loader):
                if i >= 200: # 200 samples
                    break
                img, label = img.to(device), label.to(device)

                # Targeted attack: random target
                if 'Targeted' in attack:
                    target = label
                    while target == label:
                        target = torch.randint(0, 10, (1,)).to(device)

                # attacks
                if attack == 'FGSM_Targeted':
                    adv_img = fgsm_targeted(dataset, model, img, target, eps)

                elif attack == 'FGSM_Untargeted':
                    adv_img = fgsm_untargeted(dataset, model, img, label, eps)

                elif attack == 'PGD_Targeted':
                    adv_img = pgd_targeted(dataset, model, img, target, k, eps, eps_step)

                elif attack == 'PGD_Untargeted':
                    adv_img = pgd_untargeted(dataset, model, img, label, k, eps, eps_step)


                # evaluation
                with torch.no_grad():
                    _, pred_orig = torch.max(model(normalize(img, dataset)), 1)
                    _, pred_adv = torch.max(model(normalize(adv_img, dataset)), 1)

                    if 'Targeted' in attack:
                        if pred_adv == target:
                            success += 1
                    else:
                        if pred_adv != label:
                            success += 1

                    # 5개 시각화
                    if i in [30, 70, 100, 130, 170]:
                        visualize_set.append({
                            'orig': img.cpu().squeeze(),
                            'adv': adv_img.cpu().squeeze(),
                            'orig_label': label.item(),
                            'adv_pred': pred_adv.item()
                        })

            print(f"[{dataset}] {attack}_{eps} Success Rate: {success / 200 * 100:.2f}%")
            save_plot(visualize_set, dataset, attack, eps)

# save results PNG
def save_plot(samples, dataset, attack, eps):
    fig, axes = plt.subplots(5, 3, figsize=(10, 15))
    for i, s in enumerate(samples):
        # tensor -> numpy
        orig, adv = s['orig'].numpy(), s['adv'].numpy()
        if dataset == 'CIFAR-10':
            orig, adv = np.transpose(orig, (1, 2, 0)), np.transpose(adv, (1, 2, 0))

        # Original img
        axes[i, 0].imshow(orig, cmap='gray' if dataset == 'MNIST' else None)
        axes[i, 0].set_title(f"Original ({s['orig_label']})")
        axes[i, 0].axis('off')

        # Adversarial img
        axes[i, 1].imshow(adv, cmap='gray' if dataset == 'MNIST' else None)
        axes[i, 1].set_title(f"Adv (Pred: {s['adv_pred']})")
        axes[i, 1].axis('off')

        # Perturbation img (x3)
        diff = np.clip((adv - orig) * 3, 0, 1)
        axes[i, 2].imshow(diff, cmap='gray' if dataset == 'MNIST' else None)
        axes[i, 2].set_title("Perturbation (x3)")
        axes[i, 2].axis('off')

    plt.suptitle(f"{dataset} - {attack}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"results/{dataset}_{attack}_eps{eps}.png")
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps_list = [0.05, 0.1, 0.2, 0.3]
    # 1. MNIST
    print("--- Starting MNIST Experiment ---")
    mnist_model = train_and_test(dataset='MNIST', epochs=10)
    evaluate_attack('MNIST', mnist_model, eps_list=eps_list, eps_step=0.01, k=40)

    # 2. CIFAR-10
    print("\n--- Starting CIFAR-10 Experiment ---")
    cifar_model = train_and_test(dataset='CIFAR-10', epochs=20)
    evaluate_attack('CIFAR-10', cifar_model, eps_list=eps_list, eps_step=0.01, k=40)