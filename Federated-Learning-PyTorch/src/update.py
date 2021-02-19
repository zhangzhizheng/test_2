#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.dataloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        # idxs_train = idxs[:int(1*len(idxs))]

        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        # idxs_test = idxs[:int(1*len(idxs)):]

        dataloader = DataLoader(DatasetSplit(dataset, idxs),
                                 batch_size=self.args.local_bs, shuffle=True)
        # validloader = DataLoader(DatasetSplit(dataset, idxs_val),
        #                          batch_size=int(len(idxs_val)/10), shuffle=False)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                         batch_size=int(len(idxs_test)/10), shuffle=False)
        # return trainloader, validloader, testloader
        return dataloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            #j = 0
            for batch_idx, (images, labels) in enumerate(self.dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                # j += 1
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                # if self.args.verbose and (batch_idx % 10 == 0):
                #     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         global_round, iter, batch_idx * len(images),
                #         len(self.dataloader.dataset),
                #         100. * batch_idx / len(self.dataloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # print("--------------j------------|",j)
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        j = 0
        for _, (images, labels) in enumerate(self.dataloader):
            j += 1
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        print("--------------j------------|",j)
        accuracy = correct/total
        return accuracy, loss/len(self.dataloader)


def test_inference(args, model, test_dataset, groups):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    # criterion = nn.functional.nll_loss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)
    # testloader = DataLoader(DatasetSplit(test_dataset, groups[0]),batch_size=128,shuffle=False) # test non-IID
    # print(len(testloader.batches))
    # print(len(testloader.batches))
    # dataiter = iter(testloader)
    # a,b = dataiter.next()
    # print(len(dataiter))
    # i = 0
    for _, (images, labels) in enumerate(testloader):
        # i += 2
        images, labels = images.to(device), labels.to(device)

        # Inference
        with torch.no_grad():
            outputs = model(images)
        batch_loss = nn.functional.nll_loss(outputs, labels)
        print("batchloss",batch_loss)
        loss += batch_loss.item()
        print("loss",loss)
        # Prediction
        preds = outputs.max(1)[1].type_as(labels) #by pygcn maker
        correct_batch = preds.eq(labels).double()
        # print(correct)
        correct_batch = correct_batch.sum()
        # print(correct)
        total += len(labels)
        correct += correct_batch.item()
        # print(correct)
        # _, pred_labels = torch.max(outputs, 1) # by sb
        # pred_labels = pred_labels.view(-1)
        # correct += torch.sum(torch.eq(pred_labels, labels)).item()
        # total += len(labels)
    # print(i)
    accuracy = correct/total
    print(accuracy, loss/len(testloader))
    return accuracy, loss/len(testloader)
