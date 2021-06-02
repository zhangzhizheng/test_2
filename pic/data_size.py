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

name_list = ['ResNet18','ResNet34','VGG20','MobileNetv2']

num_list = [499.061, 498.519, 301.692, 483.942]
num_list1 = [566.461, 749.055, 329.823, 551.939]
x =list(range(len(num_list)))
total_width, n = 0.5, 2
width = total_width / n
plt.figure()
# plt.title('Training Loss vs Communication rounds')
for i in range(len(x)):
    x[i] = x[i] - width/2
plt.bar(x, num_list, width=width, label='GTX3080', color='blue')
for i in range(len(x)):
    x[i] = x[i] + width
plt.plot(name_list,[0,0,0,0])
plt.bar(x, num_list1, width=width, label='GTX2080', color='green')
plt.ylabel('latency (s)')
plt.legend()
#plt.ylabel('time')
#plt.xlabel('models')
plt.savefig('computation_time.png')
