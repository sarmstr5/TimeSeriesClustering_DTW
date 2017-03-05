from heapq import nsmallest, nlargest
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from sys import stdout
from random import shuffle
from datetime import datetime as dt
from threading import Thread
from multiprocessing.pool import ThreadPool
from multiprocessing import Process, cpu_count, Pool, Queue
import json
import math
from itertools import chain
import concurrent
import traceback

def get_time():
    time = dt.now()
    hour, minute, second = str(time.hour), str(time.minute), str(time.second)
    if (len(minute) == 1):
        minute = '0' + minute
    if (len(hour) == 1):
        hour = '0' + hour
    if (len(second) ==1):
        second = '0' + second
    time = hour + minute + '.' + second
    return time


class timeseries_kNN(object):


    def __init__(self, k=1, data=None, labels=None, verbose=True):
        self.k = k
        self.data = None
        self.train_labels = None
        self.test_labels = None
        self.train_df  = None
        self.test_df = None
        self.classes_set = None
        self.verbose = verbose
        self.predictions = []
        self.distance_metric = None


    def fit(self, x, labels=None, dist_metric='euclidean'):
        # check if test_set is pandas
        if self.verbose: print('In fit: {}'.format(get_time()))

        if not isinstance(x, pd.DataFrame) or not isinstance(labels, pd.DataFrame):
            raise TypeError("Pandas Dataframe must be used for dataframes")

        self.data = x
        self.train_labels = labels
        self.train_labels.astype(int)
        labels_list = labels.transpose().values.tolist()[0]
        self.classes_set = set(labels_list) # used for voting

        self.distance_metric = dist_metric.lower()


    def predict(self, test_set, parallel=False, n_subprocesses=cpu_count()-1):
        '''
        Worker class that runs kNN
        Can run in parallel or in series
        Would like to update to use Dask in stead of Queue class

        :type test_set: object
        :param test_set:
        :param parallel:
        :return:
        '''
        self.test_df = test_set
        verbose = self.verbose
        # self.test_df.insert(loc=self.test_df.shape[1], value=np.nan, column='prediction')

        # time series i is a tuple (index, uid, movieid)
        # ISSUE - there is sometimes an issue with missing data when num_of_chunks is greater than cpus
        # ISSUE - not always an issue but undetermined bug in the software
        if parallel:
            num_of_chunks = n_subprocesses
            n_test_cases =self.test_df.shape[0]
            if num_of_chunks> cpu_count():
                print("There might be an error due to number of subprocesses")

            chunks = self.work_chunks(self.test_df, num_of_chunks)
            if verbose:
                print("Classifying in parallel with {} subprocesses;\ttime: {}".format(num_of_chunks, get_time()))
                print("The index slices are:\n{}".format(chunks))
                print("The number of test cases are: {}".format(n_test_cases))
            # creating pool of workers
            p = Pool(num_of_chunks)

            # runs kNN predictions in parallel
            # results_zipped is a list of zipped objects representing each chunnk
            # Each pool returns a tuple of df_indices and predictions, aka (df index_i, yhat_i)
            # When i join the pools I have a tuple of tuples of tuples;
            # aka tuple of pool results, index and prediction, ((pool_i), (pool_i+1), ...)
            results_zipped = p.map(self.kNN, chunks)
            p.close()
            p.join()

            # I convert those to a flat array of tuples ordered by their df_index
            # job has been mapped now processing results
            if verbose:
                print('Converting tuples to list of predictions: {}'.format(get_time()))

            # Converts to list from generator
            tup_of_tups_list = list(results_zipped)

            if verbose:
                print('Flattening to list of tuples: {}'.format(get_time()))
            # Flatten to a list of tuples
            tup_list = list(chain.from_iterable(tup_of_tups_list))

            if verbose:
                print('Sorting tuples: number of tuples: {}, time: {}'.format(len(tup_list), get_time()))

            # Sorts tuples to original sequence, aka by index ((index 1, yhat1), ...)
            # May not need to sort, but as a precaution
            sorted_tuples_yhat = sorted(tup_list, key=lambda tup: tup[0])

            if verbose:
                print('Extracting predictions: {}'.format(get_time()))

            # Extracts predictions from tuples [yhat1, yhat2, ...]
            prediction_list = [yhat for i, yhat in sorted_tuples_yhat]

            # Making sure I have the right number of predictions
            n_predictions = len(prediction_list)
            assert n_predictions == n_test_cases, "Number of predictions: {}\tNumber of tests: {}\n{}".format(
                n_predictions, n_test_cases, sorted_tuple_yhat)

            return prediction_list

        else:
            # single slice of all indices
            # Returns one ordered tuple of index and results
            # More simple and no need to organize results

            results = self.kNN(slice(0, self.test_df.shape[0]))
            results_list = [yhat for i, yhat in results]
            return results_list

    # Having issues with some variations in number of chunks leads to missing data
    # I havent had an issue when number of chunks is equal to or less than number of CPUs
    def work_chunks(self, df, n_chunks):
        num_rows = len(df)
        steps = math.ceil(num_rows/n_chunks) #rounding errors causes out of bounds
        return [slice(n, n+steps) for n in range(0, num_rows, steps)]  #should cover uneven steps

    def kNN(self, a_chunk):
        '''
        Runs kNN on chunk of test set

        :param a_chunk:
        :return:
        '''
        if self.verbose: print('Running kNN')

        predictions = []
        prediction_id_list = []
        df = self.test_df.iloc[a_chunk] # partition of test_df
        if self.verbose:
            # print("On Chunk: {}".format(a_chunk))
            print("The shape of df is {}".format(df.shape))
        i = 0
        #name is None because of pickling issue with parallel
        for test_i in df.itertuples(name=None):
            # if self.verbose:
            #     print(test_i)

            try:
                # pandas.Series of distances between xi and yi
                # self.dist(row from data based on axis, index of yi, apply along rows)
                # (xi, yi_index, rows or columns)
                distances = self.data.apply(lambda row: self.dist(row, test_i[0]), axis=1) # to each row, can be quicker

                # using euclidean distance, returns list of tuples
                neighbors = nsmallest(self.k, distances.iteritems(), key=lambda x: x[1])
                assert len(neighbors)>0, 'No nearest neightbors were found'

                yhat = self.predict_yhat(neighbors)

                if test_i[0] % 10 == 0 and self.verbose:
                    print("chunk slice: {};\ttime is:{};\tindex: {};\ti: {};\tnumber of records: {};\t{}% complete; "
                          "\t yhat: {};".
                          format(a_chunk, get_time(), test_i[0], i, len(df), round(i / len(df), 2) * 100, yhat))
                predictions.append(yhat)
                prediction_id_list.append(test_i[0])
                i += 1

            except Exception:
                print('Exception:\n{}'.format(traceback.format_exc()))
                print("chunk slice:{};\tindex:{};\ti:{};\tnumber of records:{}".format(a_chunk, test_i[0], i, len(df)))
        return zip(prediction_id_list, predictions)

    def predict_yhat(self, neighbors):
        # assumes no 0 class but shouldnt matter
        # assuming no skipped classes e.g. 4,6
        weighted_votes = np.zeros(len(self.classes_set)+1)  # no 0 class


        for neighbor in neighbors:
            index = neighbor[0]
            # getting an error here single positional index is out of bounds
            # means must be an issue with neighbor or labels dataframe
            nbor_class = int(self.train_labels.iloc[index].values[0])
            vote = 1/neighbor[1]  # smaller the distance the larger the vote
            weighted_votes[nbor_class] += vote

        voted_class = weighted_votes.argmax() #returns index(class)
        return voted_class

    def dist(self, x, y_index):
        y = self.test_df.loc[y_index,:] # row xi
        if self.distance_metric == 'euclidean':
            dist = np.sqrt( np.sum((x-y)*(x-y)) )  # sqrt( sum( (xi-yi)^2) )
        elif type == 'dtw':
            pass
        else:
            raise ValueError('Choose a distance metric of either euclidean or dtw')
        return dist
