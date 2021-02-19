import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Get_Loader(object):
    def __init__(self, args, dataset, idxs_users):
        self.args = args
        self.dataloader = self.train_val_test(dataset, list(idxs_users))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)
        
    def cifar_noniid(dataset, num_users, args):
        """
        Sample non-I.I.D client data from CIFAR10 dataset
        :param dataset:
        :param num_users:
        :return:
        """
        num_shards, num_imgs = 200, 250
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([]) for i in range(args.num_users)}
        idxs = np.arange(num_shards*num_imgs)
        # labels = dataset.train_labels.numpy()
        labels = np.array(dataset.targets)

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        if(args.data_distribution == 1):                    # Non-IID add
            rand_set_all = [1, 20 ,40 ,60 ,80 , 100, 120]
            k = [10, 20, 10, 5 ,3 ,1, 1]
        if(args.data_distribution == 2):
            rand_set_all = [180, 160, 140, 120 ,100, 80, 60]
            k = [10, 20, 10, 5 ,3 ,1, 1]
        if(args.data_distribution == 3):
            rand_set_all = [1,2]
            k = [1,2]
        if(args.data_distribution == 4):
            rand_set_all = [{180, 160, 140, 120 ,100, 80, 60}, {1, 20 ,40 ,60 ,80 , 100, 120}]
            k = [10, 20, 10, 5 ,3 ,1, 1]

        # divide and assign
        for i in range(args.num_users):
            # rand_set = set(np.random.choice(idx_shard, 2, replace=False)) # rand set the datasets
            rand_set = set(rand_set_all[i]) # 10 client static datasets
            # print(rand_set)
            # idx_shard = list(set(idx_shard) - rand_set_all[i])
            # print(rand_set_all[i])
            # for rand, j in zip(rand_set_all, k):
            for rand, j in zip(rand_set, k):
                # print(rand, j)
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+j)*num_imgs]), axis=0)
            # print(len(dict_users[0]))
        return dict_users

    def cifar_noniid_test(dataset, args):
        """
        Sample non-I.I.D client data from MNIST dataset
        :param dataset:
        :param num_users:
        :return:
        """
        # 60,000 training imgs -->  200 imgs/shard X 300 shards
        num_shards, num_imgs = 100, 100
        # idx_shard = [i for i in range(num_shards)]
        dict_users_1 = {i: np.array([]) for i in range(args.num_users)} 
        dict_users_2 = {i: np.array([]) for i in range(args.num_users)}
        idxs = np.arange(num_shards*num_imgs)
        # labels = dataset.targets.numpy()
        labels = np.array(dataset.targets)
        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]
        # print(idxs)
        # rand_set_all = [
        #     {1, 2},
        #     {3, 4},
        #     {130, 150},
        #     {29, 7},
        #     {81, 51},
        #     {73, 166},
        #     {4, 60},
        #     {115, 183},
        #     {117, 198},
        #     {177, 35},
        # # ]  #  static datasets
        # if(args.data_distribution == 1):
        #     rand_set_all = [0, 10 ,20 ,30 ,40 , 50, 60]
        # else:
        #     rand_set_all = [90, 80, 70, 60 ,50, 40, 30]

        rand_set_all_1 = [0, 10 ,20 ,30 ,40 , 50, 60]
        rand_set_all_2 = [90, 80, 70, 60 ,50, 40, 30]
        # k = [5, 10, 5, 3 ,2 ,1, 1]
        # rand_set_all = {[0,90],[10,80],[20,70],[30,60],[40,50],[50,40],[60,30]}
        dis = [5, 10, 5, 3 ,2 ,1, 1]
        # divide and assign 2 shards/client
        for i in range(args.num_users):
            # rand_set = set(np.random.choice(idx_shard, 2, replace=False)) # rand set the datasets
            # rand_set = set(rand_set_all[i]) # 10 client static datasets
            # print(rand_set)
            # idx_shard = list(set(idx_shard) - rand_set_all[i])
            # print(rand_set_all[i])
            for j in range(len(dis)):
                # print(rand_1, rand_2, j)
                dict_users_1[i] = np.concatenate(
                    (dict_users_1[i], idxs[rand_set_all_1[j]*num_imgs:(rand_set_all_1[j]+dis[j])*num_imgs]), axis=0)
                dict_users_2[i] = np.concatenate(
                    (dict_users_2[i], idxs[rand_set_all_2[j]*num_imgs:(rand_set_all_2[j]+dis[j])*num_imgs]), axis=0)
            #print(dict_users)
            # print(len(dict_users_1[0]), len(dict_users_2[0]))
        return dict_users_1, dict_users_2