import numpy as np
 
# # a = np.array([100000000,0])
# # a = set(a)
# # print(a)
# b = [20,120,220,320,420,520,620,720,820,920,1030,1140,1240,1340]
# # print(b)
# # b = set(b)
# # print(b)
# # #b = a[:,[7,0,2]]   # 从索引 2 开始到索引 7 停止，间隔为 2
# # c = list(b - a)
# # print(c)

# dict_users = {i: np.array([]) for i in range(5)}
# dict_users[0] = np.concatenate((dict_users[0], b[1*2:(1+1)*2]), axis=0)
# dict_users[0] = np.concatenate((dict_users[0], b[1*2:(1+1)*2]), axis=0)
# dict_users[1] = np.concatenate((dict_users[1], b[1*2:(1+1)*2]), axis=0)
# print(dict_users)

#PLOTTING (optional)
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# Saving the objects train_loss and train_accuracy:
file_name_1 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\4_layer_0.pkl'
file_name_2 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\4_layer_1.pkl'
file_name_3 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\5_layer_0.pkl'
file_name_4 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\5_layer_1.pkl'
# file_name_5 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\1_1_50_vgg_CIFAR10.pkl'
# # file_name_6 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\2_0_50_vgg_CIFAR10.pkl'

# file_name_7 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\1_1_50_MobileNet_animals.pkl'
# file_name_8 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\1_1_50_MobileNet_caltecth.pkl'
# file_name_9 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\1_1_50_MobileNetV2_animals.pkl'
# file_name_10 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\1_1_50_MobileNetV2_caltecth.pkl'
# file_name_11 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\1_1_50_vgg_animals.pkl'
# file_name_12 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\1_1_50_vgg_caltecth.pkl'
# dir = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\'
# model_1 = '_0_100_MobileNet_CIFAR100.pkl'
# model_2 = '_0_100_MobileNetV2_CIFAR100.pkl'
# model_3 = '_0_100_vgg19_CIFAR100.pkl'
# model_4 = '_0_100_ResNet18_CIFAR100.pkl'
# model_5 = '_0_100_MobileNetTune_CIFAR100.pkl'
# with open(dir+str(99)+model_1, 'rb') as m1:
#     [list_acc_1, list_loss_1] = pickle.load(m1)
# with open(dir+str(99)+model_2, 'rb') as m2:
#     [list_acc_2, list_loss_2] = pickle.load(m2)
# with open(dir+str(99)+model_3, 'rb') as m3:
#     [list_acc_3, list_loss_3] = pickle.load(m3)
# with open(dir+str(99)+model_3, 'rb') as m4:
#     [list_acc_4, list_loss_4] = pickle.load(m4)
# with open(dir+str(99)+model_5, 'rb') as m5:
#     [list_acc_5, list_loss_5] = pickle.load(m5)
# for i in range(101,103):
#     print(i)
#     # print(dir+str(i)+model_1)
#     with open(dir+str(i)+model_1, 'rb') as m1:
#         [list_acc_1, list_loss_1] = pickle.load(m1)
#     with open(dir+str(i)+model_2, 'rb') as m2:
#         [list_acc_2, list_loss_2] = pickle.load(m2)
#     with open(dir+str(i)+model_3, 'rb') as m3:
#         [list_acc_3, list_loss_3] = pickle.load(m3)
#     with open(dir+str(i)+model_3, 'rb') as m4:
#         [list_acc_4, list_loss_4] = pickle.load(m4)
#     with open(dir+str(i)+model_5, 'rb') as m5:
#         [list_acc_5, list_loss_5] = pickle.load(m5)
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(list_loss_1)), list_loss_1, "-", label = "m1")
    # plt.plot(range(len(list_loss_2)), list_loss_2, "-", label = "m2")
    # plt.plot(range(len(list_loss_3)), list_loss_3, "-", label = "VGG19")
    # # plt.plot(range(len(list_loss_4)), list_loss_4, "-", label = "resnet")
    # plt.plot(range(len(list_loss_5)), list_loss_5, "-", label = "mt")
    # plt.legend()
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('final_cifar100_noniid_loss_v'+str(i)+'.png')
    # plt.close()
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(list_acc_1)), list_acc_1, "-", label = "m1")
    # plt.plot(range(len(list_acc_2)), list_acc_2, "-", label = "m2")
    # plt.plot(range(len(list_acc_3)), list_acc_3, "-", label = "VGG19")
    # # plt.plot(range(len(list_acc_4)), list_acc_4, "-", label = "resnet")
    # plt.plot(range(len(list_acc_5)), list_acc_5, "-", label = "mt")
    # plt.legend()
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('final_cifar100_noniid_acc_v'+str(i)+'.png')
    # plt.close()
# plt.figure()
# plt.title('Average Accuracy vs Communication rounds')
# plt.plot(range(len(list_acc_1)), list_acc_1, "-", label = "m1")
# plt.plot(range(len(list_acc_2)), list_acc_2, "-", label = "m2")
# plt.plot(range(len(list_acc_3)), list_acc_3, "-", label = "VGG19")
# plt.plot(range(len(list_acc_4)), list_acc_4, "-", label = "resnet")
# plt.plot(range(len(list_acc_5)), list_acc_5, "-", label = "mt")
# plt.legend()
# plt.ylabel('Average Accuracy')
# plt.xlabel('Communication Rounds')
# plt.savefig('final_cifar10_noniid_acc_v'+str(99)+'.png')
# plt.close()
# file_name_1 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\1_0_100_MobileNet_CIFAR10.pkl'
# file_name_2 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\1_0_100_MobileNetV2_CIFAR10.pkl'
# # file_name_3 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\1_0_50_vgg_CIFAR100.pkl'
# file_name_4 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\2_0_100_MobileNet_CIFAR10.pkl'
# file_name_5 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\2_0_100_MobileNetV2_CIFAR10.pkl'
# # file_name_6 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\2_0_50_vgg_CIFAR10.pkl'
# file_name_7 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\1_0_100_vgg19_CIFAR10.pkl'
# file_name_8 = 'H:\\paper\\P-idea-1\\test_2\\cifar-gcn-drl\\2_0_100_vgg19_CIFAR10.pkl'
with open(file_name_1, 'rb') as f1:
    # [list_acc, list_loss] = pickle.load(f1)
    [list_acc_1, list_loss_1] = pickle.load(f1)
    # print(list_loss_1, list_acc_1, list_loss_2, list_acc_2)
with open(file_name_2, 'rb') as f2:
    [list_acc_2, list_loss_2] = pickle.load(f2)
with open(file_name_3, 'rb') as f3:
    # [list_acc, list_loss] = pickle.load(f1)
    [list_acc_3, list_loss_3] = pickle.load(f3)
    # print(list_acc_2_2, list_loss_2_2)
with open(file_name_4, 'rb') as f4:
    [list_acc_4, list_loss_4] = pickle.load(f4)
    # print(list_acc_, list_loss_4)
# with open(file_name_5, 'rb') as f5:
#     # [list_acc, list_loss] = pickle.load(f1)
#     [list_acc_5, list_loss_5] = pickle.load(f5)
#     # print(list_acc_2_2, list_loss_2_2)
# # with open(file_name_6, 'rb') as f6:
# #     [list_acc_6, list_loss_6] = pickle.load(f6)
# with open(file_name_7, 'rb') as f7:
#     [list_acc_7, list_loss_7] = pickle.load(f7)
# with open(file_name_8, 'rb') as f8:
#     [list_acc_8, list_loss_8] = pickle.load(f8)
# with open(file_name_7, 'rb') as f7:
#     [list_acc_7, list_loss_7] = pickle.load(f7)
# with open(file_name_8, 'rb') as f8:
#     [list_acc_8, list_loss_8] = pickle.load(f8)
# with open(file_name_9, 'rb') as f9:
#     [list_acc_9, list_loss_9] = pickle.load(f9)
# with open(file_name_10, 'rb') as f10:
#     [list_acc_10, list_loss_10] = pickle.load(f10) 
# with open(file_name_11, 'rb') as f11:
#     [list_acc_11, list_loss_11] = pickle.load(f11)
# with open(file_name_12, 'rb') as f12:
#     [list_acc_12, list_loss_12] = pickle.load(f12) 
#Plot Loss curve
# plt.figure()
# plt.title('Training Loss vs Communication rounds')
# plt.plot(range(len(list_loss_1)), list_loss_1, "-", label = "mT_1")
# plt.plot(range(len(list_loss_2)), list_loss_2, "-", label = "m2_1")
# # plt.plot(range(len(list_loss_3)), list_loss_3, "-", label = "VGG11_1")
# plt.plot(range(len(list_loss_7)), list_loss_7, "-", label = "vgg19_1")
# plt.plot(range(len(list_loss_4)), list_loss_4, "-", label = "mT_2")
# plt.plot(range(len(list_loss_5)), list_loss_5, "-", label = "m2_2")
# # plt.plot(range(len(list_loss_6)), list_loss_6, "-", label = "VGG11_2")
# plt.plot(range(len(list_loss_8)), list_loss_8, "-", label = "vgg19_2")
# # plt.plot(range(len(list_loss_7)), list_loss_7, "-", label = "mT_animals")
# # plt.plot(range(len(list_loss_7)), list_loss_7, "-", label = "mT_animals")
# # plt.plot(range(len(list_loss_9)), list_loss_9, "-", label = "m2_animals")
# # plt.plot(range(len(list_loss_11)), list_loss_11, "-", label = "vgg_animals")
# # plt.plot(range(len(list_loss_8)), list_loss_8, "-", label = "mT_caltecth")
# # plt.plot(range(len(list_loss_10)), list_loss_10, "-", label = "m2_caltecth")
# # plt.plot(range(len(list_loss_12)), list_loss_12, "-", label = "vgg_caltecth")
# plt.legend()
# plt.ylabel('Training loss')
# plt.xlabel('Communication Rounds')
# plt.savefig('cifar10_noniid_loss_v2.png')

# # Plot Average Accuracy vs Communication rounds
plt.figure()
plt.title('Average Accuracy vs Communication rounds')
plt.plot(range(len(list_acc_1)), list_acc_1, "-", label = "4_layer_0")
plt.plot(range(len(list_acc_3)), list_acc_3, "-", label = "5_layer_0")
plt.plot(range(len(list_acc_2)), list_acc_2, "-", label = "4_layer_1")
plt.plot(range(len(list_acc_4)), list_acc_4, "-", label = "5_layer_1")
# plt.plot(range(len(list_acc_5)), list_acc_5, "-", label = "m2_2")
# plt.plot(range(len(list_acc_8)), list_acc_8, "-", label = "vgg19_2")
# # # # plt.plot(range(len(list_acc_6)), list_acc_6, "-", label = "VGG_2")
# # plt.plot(range(len(list_acc_8)), list_acc_8, "-", label = "vgg19_2")
# # plt.plot(range(len(list_acc_7)), list_acc_7, "-", label = "mT_animals")
# # plt.plot(range(len(list_acc_9)), list_acc_9, "-", label = "m2_animals")
# # plt.plot(range(len(list_acc_11)), list_acc_11, "-", label = "vgg_animals")
# # plt.plot(range(len(list_acc_8)), list_acc_8, "-", label = "mT_caltecth")
# # plt.plot(range(len(list_acc_10)), list_acc_10, "-", label = "m2_caltecth")
# # plt.plot(range(len(list_acc_12)), list_acc_12, "-", label = "vgg_caltecth")
# # plt.plot(range(len(list_acc_2)), list_acc_2, "+-", label = "a2")
# # plt.plot(range(len(train_accuracy_3)), train_accuracy_3, "r--", label = "d2-t1")
# # plt.plot(range(len(train_accuracy_4)), train_accuracy_4, "g^", label = "d2-t2")
plt.legend()
plt.ylabel('Average Accuracy')
plt.xlabel('Communication Rounds')
plt.savefig('cifar10_noniid_acc_2_v2.png')