import os
file_name_caltech = '/home/caltech/data_list'
file_name_animals = '/home/animals/data_list'
caltech_class = os.listdir('/home/caltech/')
animals_class = os.listdir('/home/animals/')
#print(caltech, animals)
with open(file_name_caltech, 'a') as caltech:
    k = 0
    for i in caltech_class:
        images = os.listdir('/home/caltech/' + i)
        for j in images:
            caltech.write(j + ' ' + k)
        k += 1
with open(file_name_animals, 'a') as animals:
    k = 0
    for i in animals_class:
        images = os.listdir('/home/animals/' + i)
        for j in images:
            animals.write(j + ' ' + k)
        k += 1