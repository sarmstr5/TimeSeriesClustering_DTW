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

def dtw_run(train_dfs, label_dfs, test_dfs,i, k_range, dtw_width_range, num_subprocesses, parallel=True, verbose=True):
    for train_df, label_df, test_df in zip(train_dfs, label_dfs, test_dfs):
        # k = 1
        s_time = get_time()
        for k in k_range:
            for dtw_width in dtw_width_range:
                if verbose: print('\n---DTW---\nDATASET: {}\tk:{}\tDTW_width:{}\ttime: {}\n---------\n'.format(
                    i, k, dtw_width, get_time()))
                print(label_df.shape)
                # print(label_df)
                class_predictions = run_kNN(train_df, label_df, test_df, k, True, dtw_width, parallel, num_subprocesses, verbose)

                if verbose: print('Results Found for dataset: {}\ttime: {}'.format(i, get_time()))
                print_results_to_csv(class_predictions, i, True, s_time, dtw_width, k)
        i += 1
        if verbose:
            print('-------Completed!DATASET{}-------'.format(i))
            print('Started at: {}\tFinished at: {}'.format(s_time, get_time()))

def euc_run(train_dfs, label_dfs, test_dfs,i, k_range, num_subprocesses, parallel=True, verbose=True):
    for train_df, label_df, test_df in zip(train_dfs, label_dfs, test_dfs):
        # k = 1
        s_time = get_time()
        for k in k_range:
            if verbose: print('\n---EUC---\nDATASET: {}\tk:{}\ttime: {}\n---------\n'.format(
                i, k,get_time()))
            print(label_df.shape)
            # print(label_df)
            class_predictions = run_kNN(train_df, label_df, test_df, k, False, None, parallel, num_subprocesses, verbose)

            if verbose: print('Results Found for dataset: {}\ttime: {}'.format(i, get_time()))
            print_results_to_csv(class_predictions, i, False, s_time, None, k)
        i += 1
        if verbose:
            print('-------Completed!DATASET{}-------'.format(i))
            print('Started at: {}\tFinished at: {}\tRun time:{}'.format(s_time, get_time(), float(get_time())-float(s_time)))

def main():
    verbose = True
    full_run = True
    do_dtw_run = True
    parallel = True
    #---------------------------#
    num_subprocesses = cpu_count()-1
    start_time = get_time()
    #---------------------------#
    i = 1
    k_range = range(1,2,1) #should do odd k's
    dtw_width_range = range(3,5,1)

    if verbose:
        print(start_time)

    if full_run:
        test_fns, train_fns, labels_fns = get_fns(verbose)
        # dfs below are generators
        train_dfs = get_dataframes(verbose, train_fns[:3])
        label_dfs = get_dataframes(verbose, labels_fns[:3])
        test_dfs = get_dataframes(verbose, test_fns[:3])

        if do_dtw_run:
            dtw_run(train_dfs, label_dfs, test_dfs,i, k_range, dtw_width_range, num_subprocesses, parallel=True, verbose=True)
        else:
            euc_run(train_dfs, label_dfs, test_dfs,i, k_range, num_subprocesses, parallel=True, verbose=True)

    else:
        print('in else')
        i = 2
        dtw_width=4
        k=1
        do_dtw_run=True
        dataset_test = 'hw1_datasets/dataset{}/test_normalized.csv'.format(i)
        test_labels = 'hw1_datasets/dataset{}/test_labels.csv'.format(i)
        dataset_train = 'hw1_datasets/dataset{}/train_normalized.csv'.format(i)
        dataset_train_labels = 'hw1_datasets/dataset{}/train_labels.csv'.format(i)
        train_df = pd.read_csv(dataset_train, index_col=0)
        test_df = pd.read_csv(dataset_test, index_col=0)
        label_df = pd.read_csv(dataset_train_labels, index_col=0)

        class_predictions = run_kNN(train_df, label_df, test_df, k, do_dtw_run, dtw_width, parallel, num_subprocesses, verbose)
        print_results_to_csv(class_predictions, i, do_dtw_run, start_time, dtw_width, k)
        print(len(class_predictions))
        # print_results_to_csv(class_predictions, 1)

    if verbose:
        print('-------Completed!-------')
        print('Started at: {}\tFinished at: {}'.format(start_time, get_time()))


if __name__ == '__main__':
    main()


