import pandas as pd
import numpy as np
import GetTime as gt
import os
from sklearn.metrics import roc_auc_score, auc, f1_score, accuracy_score, precision_recall_fscore_support

def get_tst_label_fns(full_rn=True):
    test_label_fns = []
    for i in range(1, 6):
        test_labels = 'hw1_datasets/dataset{}/test_labels.csv'.format(i)
        test_label_fns.append(test_labels)

    return test_label_fns

def get_y(i):
    fn = 'hw1_datasets/dataset{}/test_labels.csv'.format(i)
    return pd.read_csv(fn, index_col=0)

def save_kNN_metrics(i, dist, dtw_window, k, accuracy, f1_scores, f1_avg, class_counts):
    print('Printing Results')
    accuracy_output_fn = 'test_output/scores/accuracy_scores.csv'
    test_output_fn = 'test_output/scores/dataset{}.csv'.format(i)

    #accuracy append
    with open(accuracy_output_fn, 'a') as csv:
        csv.write('{}\t{}\t{}\t{}\t{:0.4f}\t{:0.4f}\n'.format(i, dist, dtw_window, k, float(accuracy), float(f1_avg)))

    # dataset metrics
    with open(test_output_fn, 'a') as csv:
        csv.write('{}\t{}\t{}\t{}\t'.format(i, dist, dtw_window, k))
        csv.write('{:0.4f}\t'.format(float(f1_avg)))
        for score in f1_scores:
            csv.write('{:0.4f}\t'.format(float(score)))
        csv.write('{:0.4f}\t'.format(float(accuracy)))
        for class_count in class_counts:
            csv.write('{}\t'.format(class_count))
        csv.write('\n')

def get_class_predictions(i, dist, dtw_window, k):
    if dist == 'DTW':
        yhat_fn = 'test_output/test_results_dataset{}_{}_{}_k{}.csv'.format(i, dist, dtw_window,k)
    else:
        yhat_fn = 'test_output/test_results_dataset{}_{}_k{}.csv'.format(i, dist, k)
    print(yhat_fn)
    return pd.read_csv(yhat_fn, header=None)

# dataset classifier w nb a# accuracy f1 c1# c2#
def find_prediction_metrics(dtw_list, k_list, dist='DTW', verbose=True):
    # Grid search crashed ONLY GOT W=5 PREDICTIONS for w in w_list:
    if dist=='DTW':
        print('in DTW')
        for dtw_w in dtw_list:
            for k in k_list:
                for i in range(1,6):    #each dataset
                    try:
                        if verbose: print('TRY - dset: {}\ttime: {}\tw: {}\tk: {}'.format(i, gt.time(), dtw_w, k))
                        y = get_y(i)
                        yhat = get_class_predictions(i, dist, dtw_w,k)
                        f1 = f1_score(y, yhat, average=None)  # returns list score [pos neg], can use weighted
                        f1_avg = f1_score(y, yhat, average='weighted')  # returns list score [pos neg], can use weighted
                        acc = accuracy_score(y, yhat)
                        class_counts = yhat[0].value_counts(sort=False)
                        save_kNN_metrics(i, dist, dtw_w, k, acc, f1, f1_avg, class_counts)
                        i += 1
                    except:
                        print('--------------')
                        print('These parameters were skipped:')
                        print('EXCEPT - dset: {}\ttime: {}\tw: {}\tk: {}'.format(i, gt.time(), dtw_w, k))
                        print('test_output/test_results_dataset{}_{}_k{}.csv'.format(i, dist, k))
                        print('--------------')
                        continue

    else:
        print('in EUC')
        for k in k_list:
            for i in range(1,6):    #each dataset
                try:
                    if verbose: print('TRY - dset: {}\ttime: {}\tk: {}'.format(i, gt.time(), k))
                    y = get_y(i)
                    yhat = get_class_predictions(i, dist, None,k)
                    f1 = f1_score(y, yhat, average=None)  # returns list score [pos neg], can use weighted
                    f1_avg = f1_score(y, yhat, average='weighted')  # returns list score [pos neg], can use weighted
                    acc = accuracy_score(y, yhat)
                    class_counts = yhat[0].value_counts(sort=False)
                    save_kNN_metrics(i, dist, np.nan, k, acc, f1, f1_avg, class_counts)
                    i += 1
                except:
                    print('--------------')
                    print('These parameters were skipped:')
                    print('EXCEPT - dset: {}\ttime: {}\tk: {}'.format(i, gt.time(), k))
                    print('test_output/test_results_dataset{}_{}_k{}.csv'.format(i, dist, k))
                    print('--------------')
                    continue

def main ():
    print(os.listdir())

    # dtw_list = [3,5,10]
    # dist_metric=  'dtw'
    dist_metrics= ('Euclidean', 'DTW')
    for dist_metric in dist_metrics:
        if dist_metric == 'DTW':
            # dtw_list = range(3,10)
            # k_list = range(1,10)
            dtw_list = range(3,5)
            k_list = range(1,2)
            find_prediction_metrics(dtw_list, k_list, dist_metric)
        else:
            k_list = range(1,2)
            # k_list = range(1,10,2)
            find_prediction_metrics(None, k_list, dist_metric)


if __name__ == '__main__':
    main()
