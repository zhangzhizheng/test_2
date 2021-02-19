import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import pickle
import random
import copy
import pandas as pd
import numpy as np
import queue
import math
import networkx as nx
import argparse
import time
# from utils import progress_bar
from tqdm import tqdm, trange
from models import *

from Get_Loader import Get_Loader
from options import args_parser

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = args_parser()
exp_details(args)

def Set_dataset(dataset):
    if dataset == 'CIFAR10':
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        parser.add_argument('--resume', '-r', action='store_true',
                            help='resume from checkpoint')
        parser.add_argument('--epoch',default=100,type=int,help='epoch')
        args = parser.parse_args()

        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='/home/test_2/cifar-10-batches-py/', train=True, download=True, transform=transform_train)
        for i in range(args.num_users):
            train_class = Get_Loader(args, trainset, i+1)
            trainloader = train_class.get_train_dataloader(trainset, args)
            # trainloader = torch.utils.data.DataLoader(
            #     trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10( root='/home/test_2/cifar-10-batches-py/', train=False, download=True, transform=transform_test)
        test_class = Get_Loader(args, testset, 1)
        if(args.iid == 1):
            testloader = test_class.get_test_dataloader_iid(testset)
            return trainloader, testloader
        else:
            testloader_d1, testloader_d2 = test_class.get_test_dataloader_niid(testset)
            return trainloader, testloader_d1, testloader_d2
            # testloader = torch.utils.data.DataLoader(
            #     testset, batch_size=100, shuffle=False, num_workers=2)

        # classes = ('plane', 'car', 'bird', 'cat', 'deer',
        #         'dog', 'frog', 'horse', 'ship', 'truck')

        # return args, trainloader, testloader
    # elif dataset == 'MNIST':  # mnist dataset unuse
    #     parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    #     parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    #     parser.add_argument('--resume', '-r', action='store_true',
    #                         help='resume from checkpoint')
    #     parser.add_argument('--epoch',default=100,type=int,help='epoch')
    #     args = parser.parse_args()

    #     # Data
    #     print('==> Preparing data..')
    #     # normalize
    #     transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    #     # download dataset
    #     trainset = torchvision.datasets.MNIST(root = "./data/",
    #                     transform=transform,
    #                     train = True,
    #                     download = True)
    #     # load dataset with batch=64
    #     trainloader = torch.utils.data.DataLoader(dataset=trainset,
    #                                         batch_size = 64,
    #                                         shuffle = True)

    #     testset = torchvision.datasets.MNIST(root="./data/",
    #                     transform = transform,
    #                     train = False)

    #     testloader = torch.utils.data.DataLoader(dataset=testset,
    #                                         batch_size = 64,
    #                                         shuffle = False)
    #     return args, trainloader, testloader
    # else:
    #     print ('Data load error!')
    #     return 0

def Set_model(net, client, args):
    print('==> Building model..')
    Model = [None for i in range (client)]
    Optimizer = [None for i in range (client)]
    if net == 'MNISTNet':
        for i in range (client):
            Model[i] = MNISTNet()
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        global_model = MNISTNet()
        return Model, global_model, Optimizer
    elif net == 'MobileNet':
        for i in range (client):
            Model[i] = MobileNet()
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = MobileNet()
        return Model, global_model, Optimizer
    elif net == 'ResNet18':
        for i in range (client):
            Model[i] = ResNet18()
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = ResNet18()
        return Model, global_model, Optimizer

def Train(model, optimizer, client, trainloader):
    criterion = nn.CrossEntropyLoss().to(device)
    # cpu ? gpu
    for i in range(client):
        model[i] = model[i].to(device)
    P = [None for i in range (client)]

    # share a common dataset
    train_loss = [0 for i in range (client)]
    correct = [0 for i in range (client)]
    total = [0 for i in range (client)]
    Loss = [0 for i in range (client)]
    time_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
            if batch_idx < 360:
                idx = (batch_idx % client)
                model[idx].train()
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer[idx].zero_grad()
                outputs = model[idx](inputs)
                Loss[idx] = criterion(outputs, targets)
                Loss[idx].backward()
                optimizer[idx].step()
                train_loss[idx] += Loss[idx].item()
                _, predicted = outputs.max(1)
                total[idx] += targets.size(0)
                correct[idx] += predicted.eq(targets).sum().item()
    time_end = time.time()
    if device == 'cuda':
        for i in range (client):
            model[i].cpu()
    for i in range (client):
        P[i] = copy.deepcopy(model[i].state_dict())

    return P, (time_end-time_start)

def Test(model, testloader):
    # cpu ? gpu
    model = model.to(device)
    P = model.state_dict()
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        indx_target = target.clone()
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        test_loss += F.cross_entropy(output, target).data
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.cpu().eq(indx_target).sum()
    test_loss = test_loss / len(testloader) # average over number of mini-batch
    accuracy = float(correct / len(testloader.dataset))
    if device == 'cuda':
        model.cpu()
    return accuracy, test_loss.item()

def Aggregate(model, client):
    P = []
    for i in range (client):
        P.append(copy.deepcopy(model[i].state_dict()))
    for key in P[0].keys():
        for i in range (client):
            if i != 0:
                P[0][key] =torch.add(P[0][key], P[i][key])
        P[0][key] = torch.true_divide(P[0][key],client)
    return P[0]


def run(dataset, net, client):
    acc_list, loss_list = [], []
    acc_list_1, loss_list_1,acc_list_2, loss_list_2 = [], [], [], []
    X, Y, Z = [], [], []
    Y1,Y2,Z1,Z2 = [], [], [], []
    if(args.iid == 1):
        trainloader, testloader = Set_dataset(dataset)
    else:
        trainloader, testloader_d1, testloader_d2 = Set_dataset(dataset)

    model, global_model, optimizer = Set_model(net, client, args)
    pbar = tqdm(range(args.epoch))
    start_time = 0
    for i in range (args.epoch):
        Temp, process_time = Train(model, optimizer, client, trainloader)
        for j in range (client):
            model[j].load_state_dict(Temp[j])
        global_model.load_state_dict(Aggregate(copy.deepcopy(model), client))
        if(args.iid == 1):
            acc, loss = Test(global_model, testloader)
            acc_list = acc_list.append(acc)
            loss_list = loss_list.append(loss)
            pbar.set_description("Epoch: %d Accuracy: %.3f Loss: %.3f Time: %.3f" %(i, acc, loss, start_time))
        else:
            acc_1, loss_1 = Test(global_model, testloader_d1)
            acc_2, loss_2 = Test(global_model, testloader_d2)
            acc_list_1 = acc_list.append(acc_1)
            loss_list_1 = loss_list.append(loss_1)
            acc_list_2 = acc_list.append(acc_2)
            loss_list_2 = loss_list.append(loss_2)
            pbar.set_description("Epoch: %d Accuracy_d1: %.3f Loss_d1: %.3f Time: %.3f" %(i, acc_1, loss_1, start_time))
            pbar.set_description("Epoch: %d Accuracy_d2: %.3f Loss_d2: %.3f Time: %.3f" %(i, acc_2, loss_2, start_time))

        for j in range (client):
            model[j].load_state_dict(global_model.state_dict())

        start_time += process_time

        # X.append(start_time)

        # if(args.idd == 1):
        #     Y.append(acc)
        #     Z.append(loss)
        #     location = '/home/test_2/cifar-gcn-drl/Test_data/FedAVG_iid.csv'
        #     dataframe = pd.DataFrame(X, columns=['X'])
        #     dataframe = pd.concat([dataframe, pd.DataFrame(Y,columns=['Y'])],axis=1)
        #     dataframe = pd.concat([dataframe, pd.DataFrame(Z,columns=['Z'])],axis=1)
        #     dataframe.to_csv(location,mode = 'w', header = False,index=False,sep=',')
        # else:
        #     Y1.append(acc_1)
        #     Y2.append(acc_2)
        #     Z1.append(loss_1)
        #     Z2.append(loss_2)
        #     location = '/home/test_2/cifar-gcn-drl/Test_data/FedAVG_niid.csv'
        #     dataframe = pd.DataFrame(X, columns=['X'])
        #     dataframe = pd.concat([dataframe, pd.DataFrame(Y1,columns=['Y1'])],axis=1)
        #     dataframe = pd.concat([dataframe, pd.DataFrame(Y2,columns=['Y2'])],axis=1)
        #     dataframe = pd.concat([dataframe, pd.DataFrame(Z1,columns=['Z1'])],axis=1)
        #     dataframe = pd.concat([dataframe, pd.DataFrame(Z1,columns=['Z2'])],axis=1)
        #     dataframe.to_csv(location,mode = 'w', header = False,index=False,sep=',')

    file_name = '/home/test_2/cifar-gcn-drl/{}_{}_{}'.format(args.num_users, args.iid, args.epochs)

    with open(file_name, 'wb') as f:
        if(args.iid == 1):
            pickle.dump([acc_list, loss_list], f)
        else:
            pickle.dump([acc_list_1, loss_list_1, acc_list_2, loss_list_2], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(loss_list_1)), loss_list_1, color='m', label = "d1_loss")
    plt.plot(range(len(loss_list_2)), loss_list_2, color='c', label = "d2_loss")
    plt.legend()
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('/home/test_2/cifar-gcn-drl/{}_{}_{}_loss.png'.format(args.num_users, args.iid, args.epochs))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(acc_list_1)), acc_list_1, color='m', label = "d1_acc")
    plt.plot(range(len(acc_list_2)), acc_list_2, color='r', label = "d2_acc")
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('/home/test_2/cifar-gcn-drl/{}_{}_{}_acc.png'.format(args.num_users, args.iid, args.epochs))
if __name__ == '__main__':
    run(dataset = 'CIFAR10', net = 'MobileNet', client = args.num_users)