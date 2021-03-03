import os
import pandas as pd

filePath = '/home/animals/'
list = ['panda','dogs','cats']
file_name_list = []
iamge_label = []
k = 0
for i in list:
        file_name_list.append(os.listdir(filePath + i))
        for j in range(len(os.listdir(filePath + i))):
                iamge_label.append(k)
        k += 1
print(len(file_name_list))
print(len(iamge_label))
dataframe = pd.DataFrame({'FileName':file_name_list,',Labels':iamge_label})
dataframe.to_csv("/home/animals/labels.csv",index=False,sep=',')