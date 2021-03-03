import os
import numpy as np
# file_name_caltech = '/home/caltech/data_list'
# file_name_animals = '/home/animals/data_list'
# caltech_class = os.listdir('/home/caltech/')
# animals_class = os.listdir('/home/animals/')
# #print(caltech, animals)
# caltech_class.remove('data_list')
# animals_class.remove('data_list')
# with open(file_name_caltech, 'a') as caltech:
#     k = 0
#     for i in caltech_class:
#         images = os.listdir('/home/caltech/' + i)
#         for j in images:
#             caltech.write('/home/caltech/' + i + '/'+ j + ' ' + str(k) + '\n')
#         k += 1
# with open(file_name_animals, 'a') as animals:
#     k = 0
#     for i in animals_class:
#         images = os.listdir('/home/animals/' + i)
#         for j in images:
#             animals.write('/home/animals/' + i + '/' + j + ' ' + str(k) + '\n')
#         k += 1

a=np.random.randint(0,2000,size=[1,10])
print(a)