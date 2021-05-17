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
import numpy as np
import matplotlib.pyplot as plt
import math
# Saving the objects train_loss and train_accuracy:
file_name_1 = 'H:\\paper\\P-idea-1\\test_2\\time\\time_batch_32.pkl'
file_name_2 = 'H:\\paper\\P-idea-1\\test_2\\time\\time_batch_64.pkl'
file_name_3 = 'H:\\paper\\P-idea-1\\test_2\\time\\time_batch_128.pkl'
file_name_4 = 'H:\\paper\\P-idea-1\\test_2\\time\\time_batch_256.pkl'
file_name_5 = 'H:\\paper\\P-idea-1\\test_2\\time\\time_batch_512.pkl'


with open(file_name_1, 'rb') as f1:
    [time_1] = pickle.load(f1)
with open(file_name_2, 'rb') as f2:
    [time_2] = pickle.load(f2)
with open(file_name_3, 'rb') as f3:
    [time_3] = pickle.load(f3)
with open(file_name_4, 'rb') as f4:
    [time_4] = pickle.load(f4)
with open(file_name_5, 'rb') as f5:
    [time_5] = pickle.load(f5)
# with open(file_name_2, 'rb') as f2:
#     [list_acc_1, list_loss_1, list_acc_2, list_loss_2] = pickle.load(f2)
# with open(file_name_3, 'rb') as f3:
#     # [list_acc, list_loss] = pickle.load(f1)
#     [list_acc_3_2, list_loss_3_2, list_acc_4_2, list_loss_4_2] = pickle.load(f3)
#     # print(list_acc_2_2, list_loss_2_2)
# with open(file_name_4, 'rb') as f4:
#     [list_acc_3, list_loss_3, list_acc_4, list_loss_4] = pickle.load(f4)


xlabels = [32,64, 128, 256, 512]
# # min_b = np.array([min(time_1), min(time_2), min(time_3), min(time_4), min(time_5)])
# # max_b = np.array([max(time_1), max(time_2), max(time_3), max(time_4), max(time_5)])
ylabels = [time_1[0]*16, time_2[0]*8,time_3[0]*4,time_4[0]*2,time_5[0]]

# xlabels = [32, 64, 128, 256]
# # min_b = np.array([min(time_1), min(time_2), min(time_3), min(time_4), min(time_5)])
# # max_b = np.array([max(time_1), max(time_2), max(time_3), max(time_4), max(time_5)])
# ylabels_1 = [0.127,0.131,0.161,0.222,0.339]
# ylabels = [(0.131-0.127)/(64-32),(0.161-0.131)/(128-64),(0.222-0.161)/(256-128),(0.339-0.222)/(512-256)]
# ylabels_2 = [0.103,0.142,0.097,0.088,0.090]

# print(ylabels)
plt.figure()

plt.title('average time on each batch')
plt.vlines(xlabels, 0.12, ylabels, linestyle="dashed")
# plt.hlines(ylabels, 0.12, xlabels, linestyle="dashed")
#plt.scatter(xlabels, ylabels, zorder=2)
plt.plot(xlabels, ylabels, marker='o', mec='r', mfc='w', label = "batch")
#plt.plot(xlabels, ylabels_2, marker='+', mec='r', mfc='w', label = "batch_layer_2")
# plt.plot(range(len(time_2)), time_2, "-", label = "batch_64")
# plt.plot(range(len(time_3)), time_3, "-", label = "batch_128")
# plt.plot(range(len(time_4)), time_4, "-", label = "batch_256")
# plt.plot(range(len(time_5)), time_5, "-", label = "batch_512")
# plt.scatter(x[0],[-0.25],s=25,c='r') # 标注最小值
# plt.scatter([1.5],[-0.25],s=25,c='r') # 标注最小值
# plt.xlim(0,None)
plt.ylim(0.12,None)
for a, b in zip(xlabels, ylabels):
    print(b)
    plt.text(a,b,b, ha='center', va='bottom', fontsize=10)
    plt.text(a,0.12,a, ha='center', va='bottom', fontsize=10)
# for a, b in zip(xlabels, ylabels_2):
#     print(b)
#     plt.text(a,b,a, ha='center', va='bottom', fontsize=10)
plt.legend()
plt.ylabel('time(one batch in each epoch')
plt.xlabel('batch size')
plt.savefig('H:\\paper\\P-idea-1\\test_2\\time\\time_test.png')



        