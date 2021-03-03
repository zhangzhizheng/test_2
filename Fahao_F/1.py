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
import os
import time
import argparse

from autodl.auto_ingestion import data_io
from autodl.utils.util import get_solution
from autodl.metrics import autodl_auc
from autodl.auto_ingestion.dataset import AutoDLDataset
from autodl.auto_models.auto_image.model import Model as ImageModel

from autodl import AutoDLDataset
from autodl.utils.util import get_solution
from autodl.metrics import autodl_auc, accuracy

import autosklearn.classification
cls = autosklearn.classification.AutoSklearnClassifier()
cls.fit(X_train, y_train)
predictions = cls.predict(X_test)
# def run_single_model(model, dataset_dir, basename, time_budget=1200, max_epoch=50):
#     D_train = AutoDLDataset(os.path.join(dataset_dir, basename, "train"))
#     D_test = AutoDLDataset(os.path.join(dataset_dir, basename, "test"))
#     solution = get_solution(solution_dir=dataset_dir)

#     start_time = int(time.time())
#     for i in range(max_epoch):
#         remaining_time_budget = start_time + time_budget - int(time.time())
#         model.fit(D_train.get_dataset(), remaining_time_budget=remaining_time_budget)

#         remaining_time_budget = start_time + time_budget - int(time.time())
#         y_pred = model.predict(D_test.get_dataset(), remaining_time_budget=remaining_time_budget)

#         # Evaluation.
#         nauc_score = autodl_auc(solution=solution, prediction=y_pred)
#         acc_score = accuracy(solution=solution, prediction=y_pred)

#         print("Epoch={}, evaluation: nauc_score={}, acc_score={}".format(i, nauc_score, acc_score))

# def convertor_image_demo():
#     raw_autoimage_datadir = f"/home/animals/"
#     autoimage_2_autodl_format(input_dir=raw_autoimage_datadir)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="tabular example arguments")
#     parser.add_argument("--input_data_path", type=str, help="path of input data")
#     args = parser.parse_args()

#     input_dir = os.path.dirname(args.input_data_path)

#     autoimage_2_autodl_format(input_dir=input_dir)

#     new_dataset_dir = input_dir + "_formatted" + "/" + os.path.basename(input_dir)
#     datanames = data_io.inventory_data(new_dataset_dir)
#     basename = datanames[0]
#     print("train_path: ", os.path.join(new_dataset_dir, basename, "train"))

#     D_train = AutoDLDataset(os.path.join(new_dataset_dir, basename, "train"))
#     D_test = AutoDLDataset(os.path.join(new_dataset_dir, basename, "test"))

#     max_epoch = 50
#     time_budget = 1200

#     model = ImageModel(D_train.get_metadata())

#     run_single_model(model, new_dataset_dir, basename, time_budget, max_epoch)