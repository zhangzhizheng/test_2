# import os
# import pandas as pd

# filePath = '/home/animals/'
# list = ['panda','dogs','cats']
# file_name_list = []
# iamge_label = []
# k = 0
# for i in list:
#         file_name_list.append(os.listdir(filePath + i))
#         for j in range(len(os.listdir(filePath + i))):
#                 iamge_label.append(k)
#         k += 1
# file_name_list = eval('[%s]'%repr(file_name_list).replace('[', '').replace(']', ''))
# print(len(file_name_list))
# print(len(iamge_label))
# dataframe = pd.DataFrame({'FileName':file_name_list,',Labels':iamge_label})
# dataframe.to_csv("/home/animals/labels.csv",index=False,sep=',')

from autodl.convertor.image_to_tfrecords import autoimage_2_autodl_format

def convertor_image_demo():
    str = '/home'
    raw_autoimage_datadir = f"/home/animals/"
    autoimage_2_autodl_format(input_dir=raw_autoimage_datadir)

convertor_image_demo()