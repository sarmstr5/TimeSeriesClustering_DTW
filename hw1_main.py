import numpy as np
import pandas as pd
from timeseries_kNN import timeseries_kNN
from datetime import datetime as dt
from multiprocessing import Pool, cpu_count


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

def get_dataframes(verbose, fn_arr):
    if verbose:
        print('In get_dataframes: {}'.format(get_time()))
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

def print_results_to_csv(predictions, dataset_num, dtw_run, start_time, dtw_width, k):
    print('Printing Results')
    if dtw_run:
        dist_type = 'DTW'
        test_output_fn = 'test_output/test_results_dataset{}_{}_{}_k{}.csv'.format(dataset_num, dist_type, dtw_width, k)
    else:
        dist_type = 'Euclidean'
        test_output_fn = 'test_output/test_results_dataset{}_{}_k{}.csv'.format(dataset_num, dist_type, k)

    with open(test_output_fn, 'w') as results:
        for y in predictions:
            results.write('{0}\n'.format(y))

def run_kNN(train_df, train_labels, test_df, k=1, dtw_run = False, width = 10, parallel = True, nprocesses=cpu_count(), verbose=False):
    if dtw_run:
        dist_metric = 'dtw'

    else: dist_metric = 'euclidean'

    if verbose: print('Instantiating kNN')
    kNN = timeseries_kNN()

    if verbose: print('Fitting kNN')
    kNN.fit(train_df, train_labels, dist_metric, width, k)  #initializes kNN object

    return kNN.predict(test_df, parallel, nprocesses)

def main():
    verbose = True
    full_run = True
    dtw_run = True
    parallel = True
    #---------------------------#

    num_subprocesses = cpu_count()
    dtw_width = 3
    start_time = get_time()

    if verbose:
        print(start_time)

    if full_run:
        test_fns, train_fns, labels_fns = get_fns(verbose)
        # dfs below are generators
        train_dfs = get_dataframes(verbose, train_fns[1:])
        label_dfs = get_dataframes(verbose, labels_fns[1:])
        test_dfs = get_dataframes(verbose, test_fns[1:])

        i = 2
        results_array = []
        for train_df, label_df, test_df in zip(train_dfs, label_dfs, test_dfs):
            # k = 1
            s_time = get_time()
            for k in range(1,3):
                for dtw_width in range(3, 6):
                    if verbose: print('\n---------\nDATASET: {}\tk:{}\tDTW_width:{}\ttime: {}\n---------\n'.format(
                        i, k, dtw_width, get_time()))
                    print(label_df.shape)
                    # print(label_df)
                    class_predictions = run_kNN(train_df, label_df, test_df, k, dtw_run, dtw_width, parallel, num_subprocesses, verbose)

                    if verbose: print('Results Found for dataset: {}\ttime: {}'.format(i, get_time()))
                    print_results_to_csv(class_predictions, i, dtw_run, s_time, dtw_width, k)
            i += 1
            if verbose:
                print('-------Completed!{}-------'.format(i))
                print('Started at: {}\tFinished at: {}'.format(s_time, get_time()))

    else:
        print('in else')
        dataset1_test = 'hw1_datasets/dataset1/test_normalized.csv'
        test1_labels = 'hw1_datasets/dataset1/test_labels.csv'
        dataset1_train = 'hw1_datasets/dataset1/train_normalized.csv'
        dataset1_train_labels = 'hw1_datasets/dataset1/train_labels.csv'
        train1 = pd.read_csv(dataset1_train, index_col=0)
        test1 = pd.read_csv(dataset1_test, index_col=0)
        train1_labels = pd.read_csv(dataset1_train_labels, header=None, index_col=0)

        class_predictions = run_kNN(train1, train1_labels, test1, dtw_run, dtw_width, parallel, num_subprocesses, verbose)
        if verbose: print('Results Found')
        print(len(class_predictions))
        # print_results_to_csv(class_predictions, 1)

    if verbose:
        print('-------Completed!-------')
        print('Started at: {}\tFinished at: {}'.format(start_time, get_time()))


if __name__ == '__main__':
    main()


