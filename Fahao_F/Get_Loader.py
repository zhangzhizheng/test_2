import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random
import time, os, pickle
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
class Get_Loader(object):
    def __init__(self, args, train_dataset, test_dataset, idxs_users):
        self.args = args
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.idxs_users = idxs_users
        self.num_users = args.num_users
        # self.dataloader = self.train_val_test(dataset, list(idxs_users))
        # self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        # self.criterion = nn.NLLLoss().to(self.device)
    def get_dataloader(self):
        if(self.args.iid == 1):
            train_loader = torch.utils.data.DataLoader(self.train_dataset,  batch_size = 128,shuffle=False)
            test_loader = torch.utils.data.DataLoader(self.test_dataset,  batch_size = 128, shuffle=False)
        if(self.args.iid == 0):
            train, test = self.cifar_noniid()
            train_loader = {i: np.array([]) for i in range(self.args.num_users)}
            # test_loader = {i: np.array([]) for i in range(self.args.num_users)}
            # num_train = len(self.train_dataset)
            # indices = list(range(num_train))
            # split = int(np.floor(0.5 * num_train))  # split index
            for i in range(0,self.num_users):
                train_loader[i] = torch.utils.data.DataLoader(DatasetSplit(self.train_dataset, train[i]),
                                                    # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                                    batch_size = 128, shuffle=True) # test non-IID for one data distribute
            # num_vaild = len(self.test_dataset)
            # indices = list(range(num_vaild))
            # split = int(np.floor(0.5 * num_train))  # split index
            test_loader = torch.utils.data.DataLoader(self.test_dataset,  batch_size = 128, shuffle=True)
            # for i in range(0,self.num_users):
            #     test_loader[i] = torch.utils.data.DataLoader(DatasetSplit(self.test_dataset, test[i]),
            #                                         # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            #                                         batch_size = 128, shuffle=False) # test non-IID for one data distribute
        return train_loader, test_loader
        # def cifar_noniid(self):
        #     """
        #     Sample non-I.I.D client data from CIFAR10 dataset
        #     :param dataset:
        #     :param num_users:
        #     :return:
        #     """
        #     # num_shards, num_imgs = 100, 500， 初始化一些变量
        #     num_train, num_test = 50000, 10000
        #     dic_train = {i: np.array([]) for i in range(self.args.num_users)}
        #     dic_train_copy = {i: np.array([]) for i in range(self.args.num_users)}

        #     dic_test = {i: np.array([]) for i in range(self.args.num_users)}
        #     dic_test_copy = {i: np.array([]) for i in range(self.args.num_users)}

        #     idxs_train = np.arange(num_train)
        #     train_labels = np.array(self.train_dataset.targets)

        #     idxs_test = np.arange(num_test)
        #     test_labels = np.array(self.test_dataset.targets)

        #     # sort labels 排序label，没啥必要
        #     idxs_labels_train = np.vstack((idxs_train, train_labels))
        #     idxs_labels_train = idxs_labels_train[:, idxs_labels_train[1, :].argsort()]
        #     idxs_train = idxs_labels_train[0, :]
        #     labels_list_train = [[], [], [], [], [], [], [], [], [], []] # 10
        #     # labels_list_train = [[], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], []
        #     #                     ]
        #     idxs_labels_test = np.vstack((idxs_test, test_labels))
        #     idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
        #     idxs_test = idxs_labels_test[0, :]
        #     labels_list_test = [[], [], [], [], [], [], [], [], [], []]  # 10
        #     # labels_list_test = [[], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], [],
        #     # [], [], [], [], [], [], [], [], [], []
        #     # ]

        #     for i in idxs_train:
        #         labels_list_train[train_labels[i]].append(i)
        #     for i in idxs_test:
        #         labels_list_test[test_labels[i]].append(i)
        #     distribution_data = [[144,94,1561,133,1099,1466,0,0,0,0],
        #                         [327,28,264,16,354,2,100,20,200,3],
        #                         [6,6,641,1,255,4,1,2,106,1723],
        #                         [176,792,100,28,76,508,991,416,215,0],
        #                         [84,1926,1,408,133,24,771,0,0,0],
        #                         [41,46,377,541,7,235,54,1687,666,0],
        #                         [134,181,505,720,123,210,44,58,663,221],
        #                         [87,2,131,1325,1117,704,0,0,0,0],
        #                         [178,101,5,32,1553,10,163,9,437,131],
        #                         [94,125,0,147,287,100,23,217,608,279],
        #                         [379,649,106,90,35,119,807,819,3,85],
        #                         [1306,55,681,227,202,34,0,648,0,0],
        #                         [1045,13,53,6,77,70,482,7,761,494],
        #                         [731,883,15,161,387,552,4,1051,0,0],
        #                         [4,97,467,5,0,407,50,1000,1098,797],
        #                         [264,2,93,266,412,142,806,2,243,1267]
        #                         ]
        #     # distribution_data = [[144,94,156,133,109,146,0,0,0,0],
        #     #                     [327,28,264,16,354,2,100,20,200,3],
        #     #                     [6,6,641,1,255,4,1,2,106,172],
        #     #                     [176,92,100,28,76,8,91,16,15,0],
        #     #                     [84,26,1,8,133,24,71,0,0,0],
        #     #                     [41,46,77,41,7,35,54,87,66,0],
        #     #                     [134,181,5,20,23,210,44,58,63,21],
        #     #                     [87,2,131,325,117,4,0,0,0,0],
        #     #                     [178,101,5,32,53,10,163,9,37,131],
        #     #                     [94,125,0,147,87,100,23,217,8,79],
        #     #                     [9,9,106,90,35,119,7,19,3,85],
        #     #                     [6,55,1,27,2,34,0,8,0,0],
        #     #                     [45,13,53,6,77,70,2,7,61,94],
        #     #                     [1,3,15,1,7,2,4,51,0,0],
        #     #                     [4,97,67,9,0,7,50,64,98,97],
        #     #                     [4,2,93,6,12,42,6,2,43,67]
        #     #                     ]
        #     # users_list = np.random.randint(0,15,size=self.num_users) #each user gets the randomly data distribution, 16
        #     # users_list = [self.args.data_distribution] # each user gets the distribution by the paremeter
        #     for i in range(self.args.num_users):
        #         ad = 0
        #         # users_list = np.random.randint(0,15,10) # cifar100
        #         for m in range(0,10):
        #             for j in distribution_data[m]:   #0 -> i, 每个client随机  , i->0 ,改固定, 加上users_list[i]每个分配 self.args.data_distribution
        #                 for k in np.random.randint(0,len(labels_list_train[ad])-1,j):
        #                     dic_train[i] = np.insert(dic_train[i], 0, labels_list_train[ad][k])
        #                 ad += 1
        #             y = np.argsort(dic_train[i])
        #             dic_train_copy[i] = dic_train[i][y]
        #     for i in range(self.args.num_users):
        #         ad = 0
        #         for m in range(0,10):
        #             for j in distribution_data[m]:  #0 -> i, 每个client随机
        #                 for k in np.random.randint(0,len(labels_list_test[ad])-1, int(j/5)):
        #                     dic_test[i] = np.insert(dic_test[i], 0, labels_list_test[ad][k])
        #                 ad += 1
        #             y = np.argsort(dic_test[i])
        #             dic_test_copy[i] = dic_test[i][y]
        #     return dic_train_copy, dic_test_copy

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

        rand_set_all_1 = [0, 10 ,20 ,30 ,40, 50 ]
        rand_set_all_2 = [90, 80, 70, 60 ,50, 40]
        # rand_set_all_1 = [0]
        # rand_set_all_2 = [50]
        k = [5, 10, 5, 3 ,2 ,1]
        # rand_set_all = {[0,90],[10,80],[20,70],[30,60],[40,50],[50,40],[60,30]}
        # dis = [5, 10, 5, 3 ,2 ,1]
        # dis = [50]

        for j in range(len(k)):
            # print(rand_1, rand_2, j)
            dict_users_1[0] = np.concatenate(
                (dict_users_1[0], idxs[rand_set_all_1[j]*num_imgs:((rand_set_all_1[j]+k[j])*num_imgs)-1]), axis=0)
            dict_users_2[0] = np.concatenate(
                (dict_users_2[0], idxs[rand_set_all_2[j]*num_imgs:((rand_set_all_2[j]+k[j])*num_imgs)-1]), axis=0)
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

    def cifar_noniid(self):
        """
        Sample non-I.I.D client data from CIFAR10 dataset
        :param dataset:
        :param num_users:
        :return:
        """
        # num_shards, num_imgs = 100, 500， 初始化一些变量
        num_train, num_test = 50000, 10000
        dic_train = {i: np.array([]) for i in range(self.args.num_users)} # np.ones([self.args.num_users,self.args.num_classes]) # {i: np.array([]) for i in range(self.args.num_users)}
        dic_train_copy = {i: np.array([]) for i in range(self.args.num_users)}#np.ones([self.args.num_users,self.args.num_classes]) # {i: np.array([]) for i in range(self.args.num_users)}
        dic_test = {i: np.array([]) for i in range(self.args.num_users)}#np.ones([self.args.num_users,self.args.num_classes]) # {i: np.array([]) for i in range(self.args.num_users)}
        dic_test_copy = {i: np.array([]) for i in range(self.args.num_users)}#np.ones([self.args.num_users,self.args.num_classes]) #{i: np.array([]) for i in range(self.args.num_users)}

        idxs_train = np.arange(num_train)
        train_labels = np.array(self.train_dataset.targets)

        idxs_test = np.arange(num_test)
        test_labels = np.array(self.test_dataset.targets)

        # sort labels 排序label，没啥必要
        idxs_labels_train = np.vstack((idxs_train, train_labels))
        idxs_labels_train = idxs_labels_train[:, idxs_labels_train[1, :].argsort()]
        idxs_train = idxs_labels_train[0, :]
        labels_list_train = [[], [], [], [], [], [], [], [], [], []] # 10

        idxs_labels_test = np.vstack((idxs_test, test_labels))
        idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
        idxs_test = idxs_labels_test[0, :]
        labels_list_test = [[], [], [], [], [], [], [], [], [], []]  # 10

        for i in idxs_train:
            labels_list_train[train_labels[i]].append(i)
        for i in idxs_test:
            labels_list_test[test_labels[i]].append(i)
        # print(labels_list_train)
        if(self.args.status == 'b'): distribution_data = np.loadtxt("/home/test_2/Fahao_F/before.txt",delimiter=',')
        elif(self.args.status == 'as'): distribution_data = np.loadtxt("/home/test_2/Fahao_F/after_s.txt",delimiter=',')
        elif(self.args.status == 'am'): distribution_data = np.loadtxt("/home/test_2/Fahao_F/after_m.txt",delimiter=',')
        # print(distribution_data)
        print(distribution_data)
        # users_list = np.random.randint(0,15,size=self.num_users) #each user gets the randomly data distribution, 16
        # users_list = [self.args.data_distribution] # each user gets the distribution by the paremeter
        # print(self.args.num_users)
        for i in range(0,self.args.num_users):
            # ad = 0
            for m in range(0,self.args.num_classes):
                if(i == 0): 
                    dic_train[i] = np.insert(dic_train[i], 0, labels_list_train[m][0:int(distribution_data[m][i])]) 
                    # print(int(distribution_data[m][i]))
                else: 
                    dic_train[i] = np.insert(dic_train[i], 0, labels_list_train[m][int(np.sum(distribution_data[m][:i])):int(np.sum(distribution_data[m][:i]) + distribution_data[m][i])]) 
                    # print(int(distribution_data[m][i]), int(np.sum(distribution_data[m][:i])))
                # ad += int(distribution_data[m][i])
            y = np.argsort(dic_train[i])
            dic_train_copy[i] = dic_train[i][y]
            # print(len(dic_train[i]))
        for i in range(0,100):
            np.random.shuffle(dic_train_copy)
            np.random.shuffle(dic_train_copy)
            np.random.shuffle(dic_train_copy)
            np.random.shuffle(dic_train_copy)
            np.random.shuffle(dic_train_copy)
        # print(np.random.shuffle(dic_train_copy))
        for i in range(0,self.args.num_users):
            # ad = 0
            for m in range(0,self.args.num_classes):
                if(i == 0): dic_test[i] = np.insert(dic_test[i], 0, labels_list_test[m][0:int(distribution_data[m][i]/5)])
                else: dic_test[i] = np.insert(dic_test[i], 0, labels_list_test[m][int(np.sum(distribution_data[m][:i])/5):int((np.sum(distribution_data[m][:i]) + distribution_data[m][i])/5)])
                # ad += int(distribution_data[m][i])
                # print(dic_test)
            y = np.argsort(dic_test[i])
            dic_test_copy[i] = dic_test[i][y]
            print(len(dic_test[i]))
        for i in range(0,100):
            np.random.shuffle(dic_test_copy)
            np.random.shuffle(dic_test_copy)
            np.random.shuffle(dic_test_copy)
            np.random.shuffle(dic_test_copy)
            np.random.shuffle(dic_test_copy)
        # return dic_train_copy, dic_test_copy
        return dic_train, dic_test

class DatasetSplit(Dataset):
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

class MyDataset(Dataset): #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset

    def __init__(self, path, transform=None, target_transform=None): #初始化一些需要传入的参数
        super(MyDataset,self).__init__()
        fh = open(path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        random.shuffle(imgs)
        print(imgs)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
 
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # print(fn,label)
        img = Image.open(fn).convert('RGB')
        # print(img)
        # x = TF.to_tensor(img)
        # x.unsqueeze_(0)
        # print(x.shape)
        # plt.imshow(x[0])
        # m = nn.AdaptiveMaxPool2d(32)
        # print(img)
        # print("sb")
        # x = m(x)
        # print("sb")
        # print(x.shape)
        # x = x.squeeze(dim=0)
        # image = transforms.ToPILImage()(x).convert('RGB')
        # image.show()
        # if self.transform is not None:
        #     image = self.transform(image)
        # return image,label
        # print(img.shape)
        if self.transform is not None:
            img = self.transform(img)
        # print(img)
        return img,label
    def __len__(self):
        return len(self.imgs)

class ImagenetDataset(Dataset): #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset

    def __init__(self, path, train=True, transform=None, target_transform=None): #初始化一些需要传入的参数
        super(ImagenetDataset,self).__init__()
        data_file = os.path.join(path, 'train_data_batch_')
        dic_data = {}
        for i in range(1,11):
            fh = open(data_file + str(i), 'rb')
            dic = pickle.load(fh)
            dic.pop('mean')
            dic_data.update(dic)

        self.data = dic_data
        self.transform = transform
        self.target_transform = target_transform
 
    def __getitem__(self, index):
        image = self.data['data'][index]
        label = self.data['labels'][index]
        # img = Image.fromarray(np.uint8(image))
        if self.transform is not None:
            image = self.transform(image)
        print(image,label)
        return image,label
    def __len__(self):
        return len(self.data['data'])

# def load_databatch(data_folder, idx, img_size=32):
#     data_file = os.path.join(data_folder, 'train_data_batch_')
#     fh = open(data_file + str(idx), 'rb')
#     d = pickle.load(fh)
#     x = d['data']
#     y = d['labels']
#     mean_image = d['mean']

#     x = x/np.float32(255)
#     mean_image = mean_image/np.float32(255)

#     # Labels are indexed from 1, shift it so that indexes start at 0
#     y = [i-1 for i in y]
#     data_size = x.shape[0]

#     x -= mean_image

#     img_size2 = img_size * img_size

#     x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
#     x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

#     # create mirrored images
#     X_train = x[0:data_size, :, :, :]
#     Y_train = y[0:data_size]
#     X_train_flip = X_train[:, :, :, ::-1]
#     Y_train_flip = Y_train
#     X_train = np.concatenate((X_train, X_train_flip), axis=0)
#     Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

#     return dict(
#         X_train=lasagne.utils.floatX(X_train),
#         Y_train=Y_train.astype('int32'),
#         mean=mean_image)