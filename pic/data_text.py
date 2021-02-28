import os
file_name_caltech = '/home/caltech/data_list'
file_name_animals = '/home/animals/data_list'

caltech = os.listdir('/home/caltech/')
animals = os.listdir('/home/animals/')
print(caltech, animals)
# with open(file_name_101, 'a') as caltech:
# with open(file_name_animals, 'a') as animals: