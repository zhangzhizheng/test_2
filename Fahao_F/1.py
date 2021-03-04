import os
import time
import argparse
import autodl.convertor.image_to_tfrecords as a

from autodl.auto_ingestion import data_io
from autodl.utils.util import get_solution
from autodl.metrics import autodl_auc
from autodl.auto_ingestion.dataset import AutoDLDataset
from autodl.auto_models.auto_image.model import Model as ImageModel

from autodl import AutoDLDataset
from autodl.utils.util import get_solution
from autodl.metrics import autodl_auc, accuracy


def run_single_model(model, dataset_dir, basename, time_budget=1200, max_epoch=50):
    D_train = AutoDLDataset(os.path.join(dataset_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(dataset_dir, basename, "test"))
    solution = get_solution(solution_dir=dataset_dir)

    start_time = int(time.time())
    for i in range(max_epoch):
        remaining_time_budget = start_time + time_budget - int(time.time())
        model.fit(D_train.get_dataset(), remaining_time_budget=remaining_time_budget)
        print('model', model)
        remaining_time_budget = start_time + time_budget - int(time.time())
        y_pred = model.predict(D_test.get_dataset(), remaining_time_budget=remaining_time_budget)
        # print(y_pred)
        # Evaluation.
        nauc_score = autodl_auc(solution=solution, prediction=y_pred)
        acc_score = accuracy(solution=solution, prediction=y_pred)

        print("Epoch={}, evaluation: nauc_score={}, acc_score={}".format(i, nauc_score, acc_score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tabular example arguments")
    parser.add_argument("--input_data_path", type=str, help="path of input data")
    args = parser.parse_args()

    input_dir = os.path.dirname(args.input_data_path)

    a.autoimage_2_autodl_format(input_dir=input_dir)

    new_dataset_dir = input_dir + "_formatted" + "/" + os.path.basename(input_dir)
    datanames = data_io.inventory_data(new_dataset_dir)
    basename = datanames[0]
    print("train_path: ", os.path.join(new_dataset_dir, basename, "train"))

    D_train = AutoDLDataset(os.path.join(new_dataset_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(new_dataset_dir, basename, "test"))

    max_epoch = 50
    time_budget = 1200

    model = ImageModel(D_train.get_metadata())

    run_single_model(model, new_dataset_dir, basename, time_budget, max_epoch)