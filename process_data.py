# You will be given 5 time series datasets, and your job is to classify each dataset. One classic
# approach is to use 1-Nearest Neighbor (1-NN) classifier, with Euclidean Distance or Dynamic
# Time Warping as the distance measure on the raw data (i.e. no feature extraction or
# dimensionality reduction). It’s been shown that 1-NN is very competitive, and it’s one of the most
# widely used classifier for time series data. In this exercise, you will compare the classification
# accuracy of Euclidean Distance and Dynamic Time Warping. While there are many publicly
# available codes for this task, you must write your own code (including Euclidean distance, DTW,
# and 1-NN). For Dynamic Time Warping, there is a parameter w, the warping window size.
# Compare two versions of DTW: (1) no constraint on warping window size and (2) set warping
# window size to 20% of the length of the time series. The last part of the exercise is to try and
# improve the best accuracy. Some options to consider: use K-NN instead of 1-NN, learn the best
# warping window size for DTW, or something else.

# You should have 4 accuracy results for each dataset: (1) Euclidean distance, (2) DTW with no
# constraint, (3) DTW with pre-defined warping window size (20%), and (4) your choice of
# improvement.
# You can use any programming language of your choice. The only requirement is that you must
# write your own code and adhere to the honor code policy.
# Format of the datasets: Each data file contains a M-by-N matrix, where M is the number of time
# series, and N is the length of time series + 1 (the first column of the matrix contains the class
# labels). Each dataset is split into training and test set for you. Report the classification accuracy
# on the test set.
# To submit: source code, README, classification accuracy and a report (graphs showing the
# results, analysis, description of what you did, etc.)

# Develop two Disctance Metrics: Euclidean Distance and Dynamic Time Warp
# (1) no constraint on warping window size
# (2) set warping window size to 20% of the length of the time series.
# Develop a flexible nearest neighbor classifier
# Perform a grid search for the best parameters for DTW and kNN
import os
import pandas as pd
from sklearn import preprocessing

def get_fns(full_rn=True):
    if full_rn:
        dataset1_test = 'hw1_datasets/dataset1/test.txt'
        dataset1_train = 'hw1_datasets/dataset1/train.txt'
        dataset2_test = 'hw1_datasets/dataset2/test.txt'
        dataset2_train = 'hw1_datasets/dataset2/train.txt'
        dataset3_test = 'hw1_datasets/dataset3/test.txt'
        dataset3_train = 'hw1_datasets/dataset3/train.txt'
        dataset4_test = 'hw1_datasets/dataset4/test.txt'
        dataset4_train = 'hw1_datasets/dataset4/train.txt'
        dataset5_test = 'hw1_datasets/dataset5/test.txt'
        dataset5_train = 'hw1_datasets/dataset5/train.txt'
    return dataset1_test, dataset1_train, dataset2_test, dataset2_train, dataset3_test, dataset3_train, dataset4_test, dataset4_train, dataset5_test, dataset5_train,

def normalize(x, verbose=True):
    return (x - x.mean()) / x.std()

def main():
    verbose = True
    full_run = True

    test1, train1, test2, train2, test3, train3, test4, train4, test5, train5 = get_fns()
    print(os.listdir())

    # read in training dataframe
    t1_df = pd.read_csv(train1, sep='\s+', header=None)
    t2_df = pd.read_csv(train2, sep='\s+', header=None)
    t3_df = pd.read_csv(train3, sep='\s+', header=None)
    t4_df = pd.read_csv(train4, sep='\s+', header=None)
    t5_df = pd.read_csv(train5, sep='\s+', header=None)

    # first column are the cluster labels
    train1_labels = t1_df.pop(0)
    train2_labels = t2_df.pop(0)
    train3_labels = t3_df.pop(0)
    train4_labels = t4_df.pop(0)
    train5_labels = t5_df.pop(0)

    # normalize training dataframes
    t1_norm = normalize(t1_df)
    t2_norm = normalize(t2_df)
    t3_norm = normalize(t3_df)
    t4_norm = normalize(t4_df)
    t5_norm = normalize(t5_df)

    # write normalized train dataframes to disk
    t1_norm.to_csv(path_or_buf='hw1_datasets/dataset1/train_normalized.csv')
    t2_norm.to_csv(path_or_buf='hw1_datasets/dataset2/train_normalized.csv')
    t3_norm.to_csv(path_or_buf='hw1_datasets/dataset3/train_normalized.csv')
    t4_norm.to_csv(path_or_buf='hw1_datasets/dataset4/train_normalized.csv')
    t5_norm.to_csv(path_or_buf='hw1_datasets/dataset5/train_normalized.csv')

    # write training labels to disk
    train1_labels.to_csv(path='hw1_datasets/dataset1/train_labels.csv')
    train2_labels.to_csv(path='hw1_datasets/dataset2/train_labels.csv')
    train3_labels.to_csv(path='hw1_datasets/dataset3/train_labels.csv')
    train4_labels.to_csv(path='hw1_datasets/dataset4/train_labels.csv')
    train5_labels.to_csv(path='hw1_datasets/dataset5/train_labels.csv')

    # Read in test data as dataframes
    tst1_df = pd.read_csv(test1, sep='\s+', header=None)
    tst2_df = pd.read_csv(test2, sep='\s+', header=None)
    tst3_df = pd.read_csv(test3, sep='\s+', header=None)
    tst4_df = pd.read_csv(test4, sep='\s+', header=None)
    tst5_df = pd.read_csv(test5, sep='\s+', header=None)

    # Remove first column
    test1_labels = tst1_df.pop(0)
    test2_labels = tst2_df.pop(0)
    test3_labels = tst3_df.pop(0)
    test4_labels = tst4_df.pop(0)
    test5_labels = tst5_df.pop(0)

    # Normalize dataframes (x - xbar)/sigma
    tst1_norm = normalize(tst1_df)
    tst2_norm = normalize(tst2_df)
    tst3_norm = normalize(tst3_df)
    tst4_norm = normalize(tst4_df)
    tst5_norm = normalize(tst5_df)

    # write normalized dataframes to disk
    tst1_norm.to_csv(path_or_buf='hw1_datasets/dataset1/test_normalized.csv')
    tst2_norm.to_csv(path_or_buf='hw1_datasets/dataset2/test_normalized.csv')
    tst3_norm.to_csv(path_or_buf='hw1_datasets/dataset3/test_normalized.csv')
    tst4_norm.to_csv(path_or_buf='hw1_datasets/dataset4/test_normalized.csv')
    tst5_norm.to_csv(path_or_buf='hw1_datasets/dataset5/test_normalized.csv')

    # write test labels to disk
    test1_labels.to_csv(path='hw1_datasets/dataset1/test_labels.csv')
    test2_labels.to_csv(path='hw1_datasets/dataset2/test_labels.csv')
    test3_labels.to_csv(path='hw1_datasets/dataset3/test_labels.csv')
    test4_labels.to_csv(path='hw1_datasets/dataset4/test_labels.csv')
    test5_labels.to_csv(path='hw1_datasets/dataset5/test_labels.csv')

    print('This is train1: {}'.format(t1_norm.head()))
    print('This is train1 labels: {}'.format(train1_labels.head()))
    print('This is test1: {}'.format(tst1_norm.head()))
    print('This is test1 labels: {}'.format(test1_labels.head()))

if __name__ == '__main__':
    main()


