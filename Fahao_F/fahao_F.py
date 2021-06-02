import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.autograd.profiler as profiler

import pickle
import random
import copy
# import pandas as pd
import numpy as np
import queue
import math
# import networkx as nx
import argparse
import time
# from utils import progress_bar
from tqdm import tqdm
from models import *
from models import mobilenet_m2,mobilenetTune
from Get_Loader import Get_Loader, MyDataset, ImagenetDataset
from options import args_parser

from evaluation.model import NetworkCIFAR as Network
from search_space import utils

# don't remove this import
import search_space.genotypes

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Set_dataset(dataset):
    # print(dataset)
    if dataset == 'CIFAR10':
        # parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        # parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        # parser.add_argument('--resume', '-r', action='store_true',
        #                     help='resume from checkpoint')
        # parser.add_argument('--epoch',default=100,type=int,help='epoch')
        # args = parser.parse_args()

        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.5, 0.5, 0.5)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='/home/test_2/cifar-10-batches-py/', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10( root='/home/test_2/cifar-10-batches-py/', train=False, download=True, transform=transform_test)
        loader_class = Get_Loader(args, trainset, testset, 1)
        if(args.iid == 1):
            trainloader, testloader = loader_class.get_dataloader()
            return trainloader, testloader
        else:
            trainloader, testloader = loader_class.get_dataloader()
            return trainloader, testloader

    elif dataset == 'MNIST':  # mnist dataset unuse
        # parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        # parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        # parser.add_argument('--resume', '-r', action='store_true',
        #                     help='resume from checkpoint')
        # parser.add_argument('--epoch',default=100,type=int,help='epoch')
        # args = parser.parse_args()

        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.5), (0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.5), (0.5)),
        ])

        trainset = torchvision.datasets.MNIST(root='/home/test_2/mnist/', train=True, download=True, transform=transform_train)
        for i in range(args.num_users):
            train_class = Get_Loader(args, trainset, i+1)
            print(train_class)
            trainloader = train_class.get_train_dataloader(trainset, args)
            # trainloader = torch.utils.data.DataLoader(
            #     trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST( root='/home/test_2/mnist/', train=False, download=True, transform=transform_test)
        test_class = Get_Loader(args, testset, 1)
        if(args.iid == 1):
            testloader_d1, testloader_d2 = test_class.get_test_dataloader_niid(testset)
            testloader = test_class.get_test_dataloader_iid(testset)
            return trainloader, testloader_d1, testloader_d2, testloader
        else:
            testloader_d1, testloader_d2 = test_class.get_test_dataloader_niid(testset)
            testloader = test_class.get_test_dataloader_iid(testset)
            return trainloader, testloader_d1, testloader_d2, testloader
            # testloader = torch.utils.data.DataLoader(
            #     testset, batch_size=100, shuffle=False, num_workers=2)

        # classes = ('plane', 'car', 'bird', 'cat', 'deer',
        #         'dog', 'frog', 'horse', 'ship', 'truck')

        # return args, trainloader, testloader
    elif dataset == 'ImageNet':
        # parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        # parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        # parser.add_argument('--resume', '-r', action='store_true',
        #                     help='resume from checkpoint')
        # parser.add_argument('--epoch',default=100,type=int,help='epoch')
        # args = parser.parse_args()

        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.5, 0.5, 0.5)),
        ])

        trainset = torchvision.datasets.ImageNet(root='/home/test_2/imagenet/', train=True, download=True, transform=transform_train)
        for i in range(args.num_users):
            train_class = Get_Loader(args, trainset, i+1)
            print(train_class)
            trainloader = train_class.get_train_dataloader(trainset, args)
            # trainloader = torch.utils.data.DataLoader(
            #     trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.ImageNet( root='/home/test_2/imagenet/', train=False, download=True, transform=transform_test)
        test_class = Get_Loader(args, testset, 1)
        if(args.iid == 1):
            testloader_d1, testloader_d2 = test_class.get_test_dataloader_niid(testset)
            testloader = test_class.get_test_dataloader_iid(testset)
            return trainloader, testloader_d1, testloader_d2, testloader
        else:
            testloader_d1, testloader_d2 = test_class.get_test_dataloader_niid(testset)
            testloader = test_class.get_test_dataloader_iid(testset)
            return trainloader, testloader_d1, testloader_d2, testloader
            # testloader = torch.utils.data.DataLoader(
            #     testset, batch_size=100, shuffle=False, num_workers=2)

        # classes = ('plane', 'car', 'bird', 'cat', 'deer',
        #         'dog', 'frog', 'horse', 'ship', 'truck')

        # return args, trainloader, testloader
    elif dataset == 'caltecth':
        # print(dataset)
        #['brain', 'camera', 'lobster', 'ferry', 'lotus', 'flamingo']
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.5, 0.5, 0.5)),
        ])
        trainset=MyDataset(path = '/home/caltech/data_list', transform=transform_train)
        # print('done')
        for i in range(args.num_users):
            train_class = Get_Loader(args, trainset, i+1)
            # print(train_class)
            trainloader = train_class.get_train_dataloader(trainset, args)
            # trainloader = torch.utils.data.DataLoader(
            #     trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = MyDataset(path = '/home/caltech/data_list', transform=transform_train)
        test_class = Get_Loader(args, testset, 1)
        if(args.iid == 1):
            # testloader_d1, testloader_d2 = test_class.get_test_dataloader_niid(testset)
            testloader = test_class.get_test_dataloader_iid(testset)
            return trainloader, testloader, testloader, testloader
        else:
            # testloader_d1, testloader_d2 = test_class.get_test_dataloader_niid(testset)
            testloader = test_class.get_test_dataloader_iid(testset)
            return trainloader, testloader, testloader, testloader
            # testloader = torch.utils.data.DataLoader(
            #     testset, batch_size=100, shuffle=False, num_workers=2)

        # classes = ('plane', 'car', 'bird', 'cat', 'deer',
        #         'dog', 'frog', 'horse', 'ship', 'truck')

        # return args, trainloader, testloader
    elif dataset == 'animals':
        # ['panda','dogs', 'cats']
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.5, 0.5, 0.5)),
        ])
        trainset=MyDataset(path = '/home/animals/data_list', transform=transform_train)
        # print('done')
        for i in range(args.num_users):
            train_class = Get_Loader(args, trainset, i+1)
            # print(train_class)
            trainloader = train_class.get_train_dataloader(trainset, args)
            # trainloader = torch.utils.data.DataLoader(
            #     trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = MyDataset(path = '/home/animals/data_list', transform=transform_train)
        test_class = Get_Loader(args, testset, 1)
        if(args.iid == 1):
            # testloader_d1, testloader_d2 = test_class.get_test_dataloader_niid(testset)
            testloader = test_class.get_test_dataloader_iid(testset)
            return trainloader, testloader, testloader, testloader
        else:
            # testloader_d1, testloader_d2 = test_class.get_test_dataloader_niid(testset)
            testloader = test_class.get_test_dataloader_iid(testset)
            return trainloader, testloader, testloader, testloader
            # testloader = torch.utils.data.DataLoader(
            #     testset, batch_size=100, shuffle=False, num_workers=2)

        # classes = ('plane', 'car', 'bird', 'cat', 'deer',
        #         'dog', 'frog', 'horse', 'ship', 'truck')

        # return args, trainloader, testloader
    elif dataset == 'imagenet':
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            
            # transforms.Resize((32,32)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.5, 0.5, 0.5)),
        ])
        # trainset=torch.utils.data.Dataset(path = '/home/', transform=transform_train) 
          
        trainset = load_databatch('/home/Imagenet32', 1)
        # ImagenetDataset(path = '/home/', transform=transform_train)
        # for i in range(args.num_users):
        #     train_class = Get_Loader(args, trainset, i+1)
        #     # print(train_class)
        #     trainloader = train_class.get_train_dataloader(trainset, args)
        #     # trainloader = torch.utils.data.DataLoader(
        #     #     trainset, batch_size=128, shuffle=True, num_workers=2)
        # testset = ImagenetDataset(path = '/home/', transform=transform_train)
        # test_class = Get_Loader(args, testset, 1)
        # if(args.iid == 1):
        #     # testloader_d1, testloader_d2 = test_class.get_test_dataloader_niid(testset)
        #     testloader = test_class.get_test_dataloader_iid(testset)
        #     return trainloader, testloader, testloader, testloader
        # else:
        #     # testloader_d1, testloader_d2 = test_class.get_test_dataloader_niid(testset)
        #     testloader = test_class.get_test_dataloader_iid(testset)
        #     return trainloader, testloader, testloader, testloader
        #     # testloader = torch.utils.data.DataLoader(
        #     #     testset, batch_size=100, shuffle=False, num_workers=2)

        # # classes = ('plane', 'car', 'bird', 'cat', 'deer',
        # #         'dog', 'frog', 'horse', 'ship', 'truck')

        # # return args, trainloader, testloader
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
    elif net == 'MobileNetV2':
        for i in range (client):
            Model[i] = MobileNetV2()
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = MobileNetV2()
        return Model, global_model, Optimizer
    elif net == 'MobileNetTune':
        for i in range (client):
            Model[i] = MobileNetTune()
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = MobileNetTune()
        return Model, global_model, Optimizer
    elif net == 'vgg11':
        for i in range (client):
            # print('vgg')
            Model[i] = VGG('VGG11')
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = VGG('VGG11')
        return Model, global_model, Optimizer
    elif net == 'MobileNetM2':
        for i in range (client):
            Model[i] = VGG('VGG20')
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = VGG('VGG20')
        return Model, global_model, Optimizer
    elif net == 'ResNet18':
        for i in range (client):
            Model[i] = ResNet18()
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = ResNet18()
        return Model, global_model, Optimizer
    elif net == 'ResNet152':
        for i in range (client):
            Model[i] = ResNet152()
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = ResNet152()
        return Model, global_model, Optimizer
    elif net == 'ResNet34':
        for i in range (client):
            Model[i] = ResNet34()
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = ResNet34()
        return Model, global_model, Optimizer
    elif net == 'ResNet101':
        for i in range (client):
            Model[i] = ResNet101()
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = ResNet101()
        return Model, global_model, Optimizer
    elif net == 'ResNet50':
        for i in range (client):
            Model[i] = ResNet50()
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = ResNet50()
        return Model, global_model, Optimizer
    elif net == 'Federated':
        Model[0] = MobileNet()
        Model[1] = MobileNetV2()
        for i in range (client):
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    elif net == 'lenet':
        for i in range (client):
            Model[i] = LeNet()
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = LeNet()
        return Model, global_model, Optimizer
def Train(model, optimizer, client, trainloader):
    criterion = nn.CrossEntropyLoss().to(device)
    # cpu ? gpu
    for i in range(client):
        model[i] = model[i].to(device)
    P = [None for i in range (client)]
    # labels_check = [0,0,0,0,0,0,0,0,0,0]
    # share a common dataset
    train_loss = [0 for i in range (client)]
    correct = [0 for i in range (client)]
    total = [0 for i in range (client)]
    Loss = [0 for i in range (client)]
    time_start = time.time()
    idx_1 = 0
    for i in range(0,client):
        for idx,(inputs, targets) in enumerate(trainloader[i]):
            # print(idx, inputs, targets)
            # for i in targets:
            #     labels_check[i] += 1
            # print(targets)
            # idx = (batch_idx % client)

            model[i].train()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer[i].zero_grad()

            # with profiler.profile(record_shapes=True) as prof:
            #     with profiler.record_function("model_inference"):

            outputs = model[i](inputs)

            # print(outputs[0], targets)
            # print(targets-4)
            Loss[i] = criterion(outputs, targets)
            Loss[i].backward()
            optimizer[i].step()
            train_loss[i] += Loss[i].item()
            _, predicted = outputs.max(1)
            total[i] += targets.size(0)
            correct[i] += predicted.eq(targets).sum().item()
            idx_1 = idx
            break
        # print(train_loss[i] / len(trainloader[i])) # average over number of mini-batch
        # print(correct[i] / len(trainloader[i].dataset))
    time_end = time.time()
    if device == 'cuda':
        for i in range (client):
            model[i].cpu()
    for i in range (client):
        P[i] = copy.deepcopy(model[i].state_dict())


    # print(labels_check)
    # time.sleep(10)
    return P, (time_end-time_start)/idx_1
def Test(model, testloader):
    # cpu ? gpu
    model = model.to(device)
    # P = model.state_dict()
    model.eval()

    # l = 0
    # a = 0
    # for i in range(0,args.num_users):
    #     test_loss = 0
    #     correct = 0
    #     for data, target in testloader[i]:
    #         indx_target = target.clone()
    #         data, target = data.to(device), target.to(device)
    #         with torch.no_grad():
    #             output = model(data)
    #         test_loss += F.cross_entropy(output, target).data
    #         pred = output.data.max(1)[1]  # get the index of the max log-probability
    #         correct += pred.cpu().eq(indx_target).sum()
    #     l += test_loss / len(testloader[i])
    #     a += correct / len(testloader[i].dataset)
    # test_loss = l / len(testloader) # average over number of mini-batch
    # accuracy = float(a / len(testloader))
    test_loss = 0
    correct = 0

    for data, target in testloader:
        # print(target)
        indx_target = target.clone()
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        # print(target-4)
        test_loss += F.cross_entropy(output, target).data
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.cpu().eq(indx_target).sum()

    test_loss = test_loss / len(testloader) # average over number of mini-batch
    accuracy = float(correct / len(testloader.dataset))

    if device == 'cuda':
        model.cpu()
    # print(accuracy, test_loss.item())
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

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def run(dataset, client, args):
    acc_list, loss_list = [], []
    acc_list_1, loss_list_1,acc_list_2, loss_list_2 = [], [], [], []
    X, Y, Z = [], [], []
    Y1,Y2,Z1,Z2 = [], [], [], []
    if(args.iid == 1):
        # print(dataset)
        trainloader, testloader = Set_dataset(dataset)
    else:
        trainloader, testloader = Set_dataset(dataset)

    # genotype = eval("search_space.genotypes.%s" % 'DARTS')
    # model = Network(16, 10, 4, False, genotype)
    model, global_model, optimizer = Set_model(args.net, client, args)
    # print('model', model[0])
    # model = torch.load('/home/test_2/Fahao_F/wandb/offline-run-20210306_060829-33a1zl9i/files/weights.pt')
    # global_model = [None for i in range (args.num_users)]
    # model = [None for i in range (args.num_users)]
    # Optimizer = [None for i in range (client)]
    # model[0] = utils.load('/home/test_2/Fahao_F/wandb/offline-run-20210307_043033-1l7lt66d/files/weights.pt')  # 4 dataset 0
    # model[0] = utils.load('/home/test_2/Fahao_F/wandb/offline-run-20210308_001408-2e23sb4e/files/weights.pt')  # 4 dataset 14
    # model[0] = utils.load('/home/test_2/Fahao_F/wandb/offline-run-20210307_045558-1ttkon4t/files/weights.pt')  # 5 dataset 1
    # model[0] = utils.load('/home/test_2/Fahao_F/wandb/offline-run-20210307_112719-251bnz32/files/weights.pt')  # 5, dataset 2
    # model[0] = utils.load('/home/test_2/Fahao_F/wandb/offline-run-20210308_034554-2a763y5p/files/weights.pt')  # 5, dataset 0-10
    # model[0] = utils.load('/home/test_2/Fahao_F/wandb/offline-run-20210308_052228-1onsngtz/files/weights.pt')  # 5, dataset 10-0
    # print('model1',type(model1))
    # model.eval()
    # global_model = model[0]
    
    # for i in range (args.epoch):
    #     pbar = tqdm(range(args.epoch))
    #     start_time = 0

    #     acc, loss = Test(global_model, testloader)
    #     acc_list.append(acc)
    #     loss_list.append(loss)
    #     pbar.set_description("Epoch: Accuracy: %.3f Loss: %.3f Time: %.3f" %(acc, loss, start_time))
    # for i in range (client):
    #     Optimizer[i] = torch.optim.SGD(model[i].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # a = model[0].state_dict()
    pbar = tqdm(range(args.epoch))
    total_time = 0
    time_list = []
    for j in pbar:
        # Temp, process_time = Train(model, optimizer, client, trainloader)
        start_time = 0
        # pbar = tqdm(range(args.epoch))
        # Temp, process_time = Train(model, optimizer, client, trainloader)
        #without distribution
        # model, process_time = Train(model, optimizer, client, trainloader)

        criterion = nn.CrossEntropyLoss().to(device)
        # cpu ? gpu
        for i in range(client):
            model[i] = model[i].to(device)
        P = [None for i in range (client)]
        # labels_check = [0,0,0,0,0,0,0,0,0,0]
        # share a common dataset
        train_loss = [0 for i in range (client)]
        correct = [0 for i in range (client)]
        total = [0 for i in range (client)]
        Loss = [0 for i in range (client)]
        time_start = time.time()
        idx_1 = 0
        for i in range(0,client):
            for idx,(inputs, targets) in enumerate(trainloader[i]):
                # print(idx, inputs, targets)
                # for i in targets:
                #     labels_check[i] += 1
                # print(targets)
                # idx = (batch_idx % client)
                # model[i].features.register_forward_hook(get_activation('features'))
                model[i].train()
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer[i].zero_grad()

                # with profiler.profile(record_shapes=True) as prof:
                #     with profiler.record_function("model_inference"):

                outputs = model[i](inputs)

                # print(outputs[0], targets)
                # print(targets-4)
                Loss[i] = criterion(outputs, targets)
                Loss[i].backward()
                optimizer[i].step()
                idx_1 = idx
                P = copy.deepcopy(model[i].state_dict())
                for name in P:
                    print(name)
                    with open('/home/test_2/time/'+args.net+'.pkl', 'wb') as f:
                        #print('a')
                        pickle.dump(P, f)
                with torch.autograd.profiler.profile(use_cuda=True,profile_memory=True) as prof:
                    outputs = model[i](inputs)
                # print(prof)
                # prof.export_chrome_trace('/home/test_2/{}.json'.format(args.batch))
                if(idx == 0):break
            # print(train_loss[i] / len(trainloader[i])) # average over number of mini-batch
            # print(correct[i] / len(trainloader[i].dataset))
        time_end = time.time()
        
        # print(activation['fc2'])

        # x = torch.randn(1, 25)
        # output = model(x)
        # for j in range (client):
        #     model[j].load_state_dict(Temp[j])
        # global_model.load_state_dict(Aggregate(model, client))

        # if(a == global_model): print("woshi shabi")
        # a = global_model.state_dict()
        # print(a)

        #without test
        # acc, loss = Test(global_model, testloader)
        # acc_list.append(acc)
        # loss_list.append(loss)
        total_time += time_end - time_start
        # total_time += time_end - time_start
        # pbar.set_description("Epoch: %d Accuracy: %.3f Loss: %.3f Time: %.3f" %(i, acc, loss, start_time))
        pbar.set_description("Epoch: %d  Time: %.3f" %(j, total_time))
        

        #without distribution
        # for j in range (0, client):
        #     model[j].load_state_dict(copy.deepcopy(global_model.state_dict()))
    # time_list.append(total_time/args.epoch) 
    # file_name = '/home/test_2/cifar-gcn-drl/{}_{}_{}_{}_{}.pkl'.format(args.data_distribution, 
    # args.iid, args.epoch, args.net, args.dataset) # 4 layer
    # file_name = '/home/test_2/cifar-gcn-drl/clients_10_labels_10_{}_1'.format(args.status)

    # time_file_name = '/home/test_2/time/time_batch_{}.pkl'.format(args.batch)

    # with open(file_name, 'wb') as f:
    #     pickle.dump([acc_list, loss_list], f)
            # pickle.dump([acc_list_1, loss_list_1], f)
    # with open(time_file_name, 'wb') as f:
    #     pickle.dump([time_list], f)
    # # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(loss_list_1)), loss_list_1, "x-", color='m', label = "d1_loss")
    # plt.plot(range(len(loss_list_2)), loss_list_2, "+-", color='c', label = "d2_loss")
    # plt.legend()
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('/home/test_2/cifar-gcn-drl/{}_{}_{}_{}_loss.png'.format(args.data_distribution, args.iid, args.epoch, args.net))

    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(acc_list_1)), acc_list_1, "x-", color='m', label = "d1_acc")
    # plt.plot(range(len(acc_list_2)), acc_list_2, "+-", color='r', label = "d2_acc")
    # plt.legend()
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('/home/test_2/cifar-gcn-drl/{}_{}_{}_{}_acc.png'.format(args.data_distribution, args.iid, args.epoch, args.net))
if __name__ == '__main__':
    args = args_parser()
    # print(args.dataset) args.dataset
    run(dataset = args.dataset, client = args.num_users, args = args)