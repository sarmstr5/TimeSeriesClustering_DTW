import pandas as pd


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

file_generator1 = (pd.read_csv(fn, index_col=0) for fn in train_fns)
file_generator2 = (pd.read_csv(fn, index_col=0) for fn in test_fns)

i=1
for train_df, test_df in zip(file_generator1, file_generator2 ):
    print('The number of TEST_{} entities is:{} and the length of the time series is:{}'.format(i, train_df.shape[0], train_df.shape[1]))
    print('The number of TRAIN_{} entities is:{} and the length of the time series is:{}'.format(i, test_df.shape[0], test_df.shape[1]))
    i+=1




