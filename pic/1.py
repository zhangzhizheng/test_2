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
file_name_1 = 'H:\\paper\\P-idea-1\\test_2\pic\\1_0_100.pkl'
#file_name_2 = 'mnist_cnn_30_C[0.1]_iid[0]_E[10]_B[10]_D2.pkl'
with open(file_name_1, 'rb') as f1:
    [list_acc_1, list_loss_1, list_acc_2, list_loss_2] = pickle.load(f1)
    print(list_loss_1, list_acc_1, list_loss_2, list_acc_2)
# with open(file_name_2, 'rb') as f2:
#     [train_loss_2, train_accuracy_3, train_accuracy_4] = pickle.load(f2)


#Plot Loss curve
plt.figure()
plt.title('Training Loss vs Communication rounds')
plt.plot(range(len(list_loss_1)), list_loss_1, "x-", label = "l1")
plt.plot(range(len(list_loss_2)), list_loss_2, "+-", label = "l2")
# plt.plot(range(len(test_loss_2)), test_loss_2, "g^", label = "t2")
plt.legend()
plt.ylabel('Training loss')
plt.xlabel('Communication Rounds')
plt.savefig('1_0_100_loss.png')

# Plot Average Accuracy vs Communication rounds
plt.figure()
plt.title('Average Accuracy vs Communication rounds')
plt.plot(range(len(list_acc_1)), list_acc_1, "x-", label = "a1")
plt.plot(range(len(list_acc_2)), list_acc_2, "+-", label = "a2")
# plt.plot(range(len(train_accuracy_3)), train_accuracy_3, "r--", label = "d2-t1")
# plt.plot(range(len(train_accuracy_4)), train_accuracy_4, "g^", label = "d2-t2")
plt.legend()
plt.ylabel('Average Accuracy')
plt.xlabel('Communication Rounds')
plt.savefig('1_0_100_acc.png')