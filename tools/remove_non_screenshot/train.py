import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models


def get_model():
    model = models.resnet50(pretrained=True)

    '''
    for param in model.parameters():
        param.requires_grad = False
    '''

    model.fc = nn.Sequential(nn.Linear(2048, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512, 2),
                             nn.LogSoftmax(dim=1))
    return model


def get_transforms():
    img_size = 224
    normalize = transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    train_transforms = transforms.Compose([transforms.Resize((img_size, img_size)),
                                           # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3),
                                           #                                  scale=(0.5, 0.75)),
                                           transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                                                                  hue=0.1),
                                           # transforms.RandomCrop((img_size,img_size), pad_if_needed=True),
                                           transforms.ToTensor(),
                                           normalize
                                           ])
    test_transforms = transforms.Compose([transforms.Resize((img_size, img_size)),
                                          transforms.ToTensor(),
                                          normalize,
                                          ])
    return train_transforms, test_transforms


def load_split_train_test(data_dir, valid_size=.2):
    train_transforms, test_transforms = get_transforms()
    train_data = datasets.ImageFolder(data_dir,
                                      transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir,
                                     transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=test_sampler, batch_size=64)
    return trainloader, testloader


CHECKPOINT_PATH = 'screen_model.pth'


def main():
    data_dir = 'data'
    trainloader, testloader = load_split_train_test(data_dir, .2)
    print(trainloader.dataset.classes)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    model = get_model()

    model.to(device)
    model.train()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    epochs = 100
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))
        print(f"Epoch {epoch + 1}/{epochs}.. "
              f"Train loss: {running_loss / print_every:.3f}.. "
              f"Test loss: {test_loss / len(testloader):.3f}.. "
              f"Test accuracy: {accuracy / len(testloader):.3f}")
        running_loss = 0
        model.train()
    torch.save(model, CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
