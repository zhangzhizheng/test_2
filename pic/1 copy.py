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
file_name_1 = 'H:\\paper\\P-idea-1\\test_2\pic\\1_0_90_MobileNet.pkl'
file_name_2 = 'H:\\paper\\P-idea-1\\test_2\pic\\2_0_90_MobileNet.pkl'
file_name_3 = 'H:\\paper\\P-idea-1\\test_2\pic\\1_0_90_MobileNetV2.pkl'
file_name_4 = 'H:\\paper\\P-idea-1\\test_2\pic\\2_0_90_MobileNetV2.pkl'


with open(file_name_1, 'rb') as f1:
    # [list_acc, list_loss] = pickle.load(f1)
    [list_acc_1, list_loss_1,_,_] = pickle.load(f1)
    # print(list_loss_1, list_acc_1, list_loss_2, list_acc_2)
with open(file_name_2, 'rb') as f2:
    [_,_,list_acc_2, list_loss_2] = pickle.load(f2)
with open(file_name_3, 'rb') as f3:
    # [list_acc, list_loss] = pickle.load(f1)
    [list_acc_3, list_loss_3,_,_] = pickle.load(f3)
    # print(list_acc_2_2, list_loss_2_2)
with open(file_name_4, 'rb') as f4:
    [_,_,list_acc_4, list_loss_4] = pickle.load(f4)
    # print(list_acc_, list_loss_4)


#Plot Loss curve
plt.figure()
plt.title('Training Loss vs Communication rounds')
plt.plot(range(len(list_loss_1)), list_loss_1, "-", label = "m1_cifar")
plt.plot(range(len(list_loss_2)), list_loss_2, "-", label = "m1_mnist")
plt.plot(range(len(list_loss_3)), list_loss_3, "-", label = "m2_cifar")
plt.plot(range(len(list_loss_4)), list_loss_4, "-", label = "m2_mnist")
plt.legend()
plt.ylabel('Training loss')
plt.xlabel('Communication Rounds')
plt.savefig('cifa12_loss_v2.png')

# Plot Average Accuracy vs Communication rounds
plt.figure()
plt.title('Average Accuracy vs Communication rounds')
# plt.plot(range(len(list_acc_3)), list_acc_3, "+-", label = "a3")
# plt.plot(range(len(list_acc_1)), list_acc_1, "x-", label = "a1")
plt.plot(range(len(list_acc_1)), list_acc_1, "-", label = "m1_cifar")
plt.plot(range(len(list_acc_2)), list_acc_2, "-", label = "m1_mnist")
plt.plot(range(len(list_acc_3)), list_acc_3, "-", label = "m2_cifar")
plt.plot(range(len(list_acc_4)), list_acc_4, "-", label = "m2_mnist")

# plt.plot(range(len(list_acc_2)), list_acc_2, "+-", label = "a2")
# plt.plot(range(len(train_accuracy_3)), train_accuracy_3, "r--", label = "d2-t1")
# plt.plot(range(len(train_accuracy_4)), train_accuracy_4, "g^", label = "d2-t2")
plt.legend()
plt.ylabel('Average Accuracy')
plt.xlabel('Communication Rounds')
plt.savefig('cifa12_acc_v2.png')