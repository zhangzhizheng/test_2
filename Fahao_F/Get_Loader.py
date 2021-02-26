import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset

import time
class Get_Loader(object):
    def __init__(self, args, dataset, idxs_users):
        self.args = args
        self.dataset = dataset
        self.idxs_users = idxs_users
        self.num_users = args.num_users
        # self.dataloader = self.train_val_test(dataset, list(idxs_users))
        # self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        # self.criterion = nn.NLLLoss().to(self.device)
    def get_train_dataloader(self, dataset, args):
        if(args.iid == 1):
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
            
        if(args.iid == 0):
            groups = self.cifar_noniid()
            # print(len(groups), len(groups[0]))
            train_loader = torch.utils.data.DataLoader(DatasetSplit(dataset, groups[0]),
                                                    batch_size=128,shuffle=False) # test non-IID for one data distribute
            print("train_loader", len(train_loader))
        return train_loader
    def get_test_dataloader_iid(self, dataset):
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        return test_loader
    def get_test_dataloader_niid(self, dataset): #non-IID dataloader
        groups_d1, groups_d2 = self.cifar_noniid_test()
        # print(groups_d1, groups_d2)

        test_loader_d1 = torch.utils.data.DataLoader(DatasetSplit(dataset, groups_d1[0]),
                                                    batch_size=128,shuffle=False)
        test_loader_d2 = torch.utils.data.DataLoader(DatasetSplit(dataset, groups_d2[0]),
                                                    batch_size=128,shuffle=False)
        print("test_loader_d1, test_loader_d2", len(test_loader_d1), len(test_loader_d2))
        return test_loader_d1, test_loader_d2
    def cifar_noniid(self):
        """
        Sample non-I.I.D client data from CIFAR10 dataset
        :param dataset:
        :param num_users:
        :return:
        """
        # num_shards, num_imgs = 100, 500
        num_shards, num_imgs = 200, 250
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([]) for i in range(self.args.num_users)}
        dict_users_copy = {i: np.array([]) for i in range(self.args.num_users)}
        idxs = np.arange(num_shards*num_imgs)
        labels = np.array(self.dataset.targets)

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]
        # for idx in idxs:
        #     print(labels[idx])
        #     time.sleep(1)
        if(self.args.data_distribution == 1):                    # Non-IID add
            # rand_set_all = [0, 20 , 40, 60, 80, 100]
            # k = [2, 4, 6, 10,20, 10]
            k = [100]
            rand_set_all = [0]
            

        if(self.args.data_distribution == 2):
            # rand_set_all = [180, 160, 140 ,120, 100, 80]
            # k = [10, 20, 10, 6, 4, 2]
            k = [100]
            rand_set_all = [100]
            # print("distribution", 2)
        if(self.args.data_distribution == 3):
            rand_set_all = [0]
            k = [200]
        if(self.args.data_distribution == 4): # the double models train together
            rand_set_all = [{180, 160, 140, 120 ,100, 80, 60}, {1, 20 ,40 ,60 ,80 , 100, 120}]
            k = [10, 20, 10, 5 ,3 ,1, 1]
            # divide and assign
            for i in range(self.args.num_users):
                rand_set = set(rand_set_all[i]) # 10 client static datasets
                for rand, j in zip(rand_set, k):
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand*num_imgs:(rand+j)*num_imgs]), axis=0)
                return dict_users
        for i in range(self.args.num_users):
            for rand, j in zip(rand_set_all, k):
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+j)*num_imgs]), axis=0)
        # print(len(dict_users), len(dict_users[0]))
        y = np.argsort(dict_users[0])
        # print(int(dict_users[0][y]))
        dict_users_copy[0] = dict_users[0][y]
        # for idx in dict_users[0]:
        #     idx = int(idx)
        #     print(labels[idx])
        #     time.sleep(0.01)
        return dict_users_copy

    def cifar_noniid_test(self):
        """
        Sample non-I.I.D client data from MNIST dataset
        :param dataset:
        :param num_users:
        :return:
        """
        num_shards, num_imgs = 100, 100 # cifa test 100x100
        # idx_shard = [i for i in range(num_shards)]
        dict_users_1 = {i: np.array([]) for i in range(self.args.num_users)} 
        dict_users_2 = {i: np.array([]) for i in range(self.args.num_users)}
        idxs = np.arange(num_shards*num_imgs)
        # labels = dataset.targets.numpy()
        labels = np.array(self.dataset.targets)
        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # rand_set_all_1 = [0, 10 ,20 ,30 ,40, 50 ]
        # rand_set_all_2 = [90, 80, 70, 60 ,50, 40]
        rand_set_all_1 = [0]
        rand_set_all_2 = [50]
        # k = [5, 10, 5, 3 ,2 ,1]
        # rand_set_all = {[0,90],[10,80],[20,70],[30,60],[40,50],[50,40],[60,30]}
        # dis = [5, 10, 5, 3 ,2 ,1]
        dis = [50]

        for j in range(len(dis)):
            # print(rand_1, rand_2, j)
            dict_users_1[0] = np.concatenate(
                (dict_users_1[0], idxs[rand_set_all_1[j]*num_imgs:((rand_set_all_1[j]+dis[j])*num_imgs)-1]), axis=0)
            dict_users_2[0] = np.concatenate(
                (dict_users_2[0], idxs[rand_set_all_2[j]*num_imgs:((rand_set_all_2[j]+dis[j])*num_imgs)-1]), axis=0)
        y_1 = np.argsort(dict_users_1[0])
        # # print(dict_users_1[0][y_1])
        dict_users_1[0] = dict_users_1[0][y_1]
        y_2 = np.argsort(dict_users_2[0])
        # # print(dict_users_2[0][y_2])
        dict_users_2[0] = dict_users_2[0][y_2]
        # # return dict_users_1, dict_users_2
        # for idx in dict_users_1[0]:
        #     idx = int(idx)
        #     print(labels[idx])
        #     time.sleep(0.01)
        # for idx in dict_users_2[0]:
        #     idx = int(idx)
        #     print(labels[idx])
        #     time.sleep(0.01)
        return dict_users_1, dict_users_2

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        # print(self.idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        sample=self.dataset[self.idxs[item]]
        # if self.transform:
        #     sample=self.transform(sample)
        # print(sample)
        return sample
        # image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
