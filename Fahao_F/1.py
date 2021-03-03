import os

filePath = '/home/animals/'
list = ['panda','dogs','cats']
file_name_list = []
iamge_label = []
k = 0
for i in list:
        file_name_list.append(os.listdir(filePath + i))
        for j in len(os.listdir(filePath + i)):
                iamge_label.append(k)
        k += 1