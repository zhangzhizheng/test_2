#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users, args):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
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
        rand_set_all = [{1, 20 ,40 ,60 ,80 , 100, 120}, {180, 160, 140, 120 ,100, 80, 60}]
        k = [10, 20, 10, 5 ,3 ,1, 1]

    # divide and assign
    for i in range(num_users):
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
        print(len(dict_users[0]))
    return dict_users

def cifar_noniid_test(dataset, num_users, args):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 100, 100
    # idx_shard = [i for i in range(num_shards)]
    dict_users_1 = {i: np.array([]) for i in range(num_users)} 
    dict_users_2 = {i: np.array([]) for i in range(num_users)}
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
    for i in range(num_users):
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
        print(len(dict_users_1[0]), len(dict_users_2[0]))
    return dict_users_1, dict_users_2

if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
