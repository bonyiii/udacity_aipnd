# Imports python modules
import argparse
import sys
from time import time, sleep
import os

import json

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

def loader(img_dir, augment = False):
    if augment:
        loader_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    else:
        loader_transforms = transforms.Compose([transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    dataset = datasets.ImageFolder(img_dir, transform = loader_transforms)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True)

    return [dataset, data_loader]


def load_model(arch, hidden_units):
    if arch == "vgg":
        model = models.vgg11(pretrained=True)
        input_units = 25088
    elif arch == "alexnet":
        model = models.alexnet(pretrained=True)
        input_units = 9216
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
        input_units = 1024

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_units, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout()),
                          ('fc3', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier

    return model


def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

def check_accuracy_on_test(model, testloader, criterion, device):
    test_loss, accuracy = validation(model, testloader, criterion, device)
    print('Accuracy of the network on the {} test images: {} %'.format(len(testloader), (accuracy / len(testloader)) * 100))


def train(model, trainloader, validationloader, epochs, learning_rate, criterion, print_every = 40, device = 'cpu'):
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    epochs = epochs
    print_every = print_every
    steps = 0

    model.to(device)

    for e in range(epochs):
        running_loss = 0
        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validationloader, criterion, device = device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(validationloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validationloader)))

                running_loss = 0

    return model

def save_checkpoint(model, train_dataset, save_dir, arch):
    model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'arch': arch
    }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(checkpoint,  save_dir + '/checkpoint.pth')
    print(checkpoint['classifier'])


def debug_args(args):
    print("Command Line Arguments:",
          "\n   arch =", args.arch,
          "\n    dir =", args.dir,
          "\n    gpu =", args.gpu,
          "\n epochs =", args.epochs,
          "\n learning_rate =", args.learning_rate)


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('dir', type = str, default = 'images', help = 'Root image data directory, train and test folders should nested underneath Eg: images/train, images/test ')
    parser.add_argument('--arch', '-a', type = str, default = 'vgg', help = 'Architecture type to use for image recognition, Default: vgg')
    parser.add_argument('--save_dir', '-s', type = str, default = 'checkpoints', help = 'Checkopints directory name, Default: chekcpoints')
    parser.add_argument('--gpu', '-g', action = 'store_true', help = 'Whether or not use GPU acceleration, Default: False')
    parser.add_argument('--epochs', type = int, default = 3, help = 'Number of epochs, Default: 3')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning rate, Default: 0.001')
    parser.add_argument('--hidden_units', type = int, default = 4096, help = 'Number of hidden units, Default: 4096')

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()

def main():
    start_time = time()
    # Creates & retrieves Command Line Arugments
    args = get_input_args()

    debug_args(args)
    train_dataset, trainloader = loader(args.dir + '/train', augment = True)
    test_dataset, testloader = loader(args.dir + '/test')
    validation_dataset, validationloader = loader(args.dir + '/valid')
    model = load_model(args.arch, args.hidden_units)

    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    criterion = nn.NLLLoss()
    train(model, trainloader, validationloader, epochs = args.epochs,  learning_rate = args.learning_rate, criterion = criterion, print_every = 40, device = device)
    check_accuracy_on_test(model, testloader, criterion, device)
    save_checkpoint(model, train_dataset, args.save_dir, args.arch)

    # Measure total program runtime by collecting end time
    end_time = time()

    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

# Call to main function to run the program
if __name__ == "__main__":
    main()
