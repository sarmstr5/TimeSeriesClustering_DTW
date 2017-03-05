import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from timeseries_kNN import timeseries_kNN
import os
from heapq import nsmallest
from random import shuffle
from datetime import datetime as dt
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, cpu_count
from threading import Thread
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from dask import delayed


def get_fns(full_rn=True):
    if full_rn:
        dataset1_test = 'hw1_datasets/dataset1/test_normalized.csv'
        dataset1_train = 'hw1_datasets/dataset1/train_normalized.csv'
        dataset2_test = 'hw1_datasets/dataset2/test_normalized.csv'
        dataset2_train = 'hw1_datasets/dataset2/train_normalized.csv'
        dataset3_test = 'hw1_datasets/dataset3/test_normalized.csv'
        dataset3_train = 'hw1_datasets/dataset3/train_normalized.csv'
        dataset4_test = 'hw1_datasets/dataset4/test_normalized.csv'
        dataset4_train = 'hw1_datasets/dataset4/train_normalized.csv'
        dataset5_test = 'hw1_datasets/dataset5/test_normalized.csv'
        dataset5_train = 'hw1_datasets/dataset5/train_normalized.csv'
        ds1_train_labels = 'hw1_datasets/dataset1/train_labels.csv'
        ds2_train_labels = 'hw1_datasets/dataset2/train_labels.csv'
        ds3_train_labels = 'hw1_datasets/dataset3/train_labels.csv'
        ds4_train_labels = 'hw1_datasets/dataset4/train_labels.csv'
        ds5_train_labels = 'hw1_datasets/dataset5/train_labels.csv'
        test_fns = [dataset1_test, dataset2_test, dataset3_test, dataset4_test, dataset5_test]
        train_fns = [dataset1_train, dataset2_train, dataset3_train, dataset4_train, dataset5_train]
        labels_fns = [ds1_train_labels, ds2_train_labels, ds3_train_labels, ds4_train_labels, ds5_train_labels]

    return test_fns, train_fns, labels_fns

def get_dataframes(verbose, fn_arr, labels=False):
    if verbose:
        print('In get_dataframes: {}'.format(get_time()))
    if labels:
        file_generator = (pd.read_csv(fn, index_col=0, header=None) for fn in fn_arr)
        # for fn in fn_arr:
        #     csv = pd.read_csv(fn, index_col=0, header=None)
        #     print(csv)
    else:
        file_generator = (pd.read_csv(fn, index_col=0) for fn in fn_arr)
    return file_generator

def get_time():
    time = dt.now()
    hour, minute, second = str(time.hour), str(time.minute), str(time.second)
    if(len(minute) == 1):
        minute = '0'+ minute
    if(len(hour) == 1):
        hour = '0' + hour
    if (len(second) ==1):
        second = '0' + second
    time = hour + minute + '.' + second
    return time

def print_results_to_csv(predictions, dataset_num):
    print('Printing Results')
    test_output_fn = 'test_output/test_results_dataset{}_{}.csv'.format(dataset_num,get_time())

    with open(test_output_fn, 'w') as results:
        for y in predictions:
            results.write('{0}\n'.format(y))

def run_kNN(train_df, train_labels, test_df, dtw_run = False, parallel = True, nprocesses=cpu_count(), verbose=False):
    if dtw_run: dist_metric='dtw'
    else: dist_metric='euclidean'

    if verbose: print('Instantiating kNN')
    kNN = timeseries_kNN()

    if verbose: print('Fitting kNN')
    kNN.fit(train_df, train_labels, dist_metric)  #initializes kNN object

    return kNN.predict(test_df, parallel, nprocesses)

def main():
    verbose = True
    full_run = True
    dtw_run = False
    parallel = True
    num_subprocesses = cpu_count()-1
    width = 1
    start_time = get_time()

    if verbose:
        print(get_time())

    if full_run:
        test_fns, train_fns, labels_fns = get_fns(verbose)
        # dfs below are generators
        train_dfs = get_dataframes(verbose, train_fns)
        label_dfs = get_dataframes(verbose, labels_fns, True)
        test_dfs = get_dataframes(verbose, test_fns)

        results_array = []
        i = 1
        for train_df, label_df, test_df in zip(train_dfs, label_dfs, test_dfs):
            print(label_df.shape)
            # print(label_df)
            class_predictions = run_kNN(train_df, label_df, test_df, dtw_run, parallel, num_subprocesses, verbose)

            if verbose: print('Results Found for dataset: {}\ttime: {}'.format(i, get_time()))
            print_results_to_csv(class_predictions, i)
            i += 1
    else:
        print('in else')
        dataset1_test = 'hw1_datasets/dataset2/test_normalized.csv'
        test1_labels = 'hw1_datasets/dataset2/test_labels.csv'
        dataset1_train = 'hw1_datasets/dataset2/train_normalized.csv'
        dataset1_train_labels = 'hw1_datasets/dataset2/train_labels.csv'
        train1 = pd.read_csv(dataset1_train, index_col=0)
        test1 = pd.read_csv(dataset1_test, index_col=0)
        train1_labels = pd.read_csv(dataset1_train_labels, header=None, index_col=0)

        class_predictions = run_kNN(train1, train1_labels, test1, dtw_run, parallel, num_subprocesses, verbose)
        if verbose: print('Results Found')
        print(len(class_predictions))
        # print_results_to_csv(class_predictions, 1)

    if verbose:
        print('-------Completed!{}-------')
        print('Started at: {}\tFinished at: {}').format(start_time, get_time()))




    # knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric='minkowski', n_jobs=-2).fit(r_df)
if __name__ == '__main__':
    main()


