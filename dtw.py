import numpy as np
from datetime import datetime as dt

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

def dtw_dist(verbose, x, y, w):
    if verbose:
        print('in dtw_dist: {}'.format(get_time()))
    # pass
    width = w
    # calculate distance matrix
    dtw_dist_mx = create_dist_mx(verbose, x, y)
    # calculate shortest path
    dist, path, iterations = shortest_path(verbose, dtw_dist_mx, w)

    return dist

def calculate_width(verbose):
    if verbose:
        print("Calculating width: {}".format(get_time()))
    pass


def create_dist_mx(verbose, x, y):
    if verbose:
        print('Creating distance Matrix: {}'.format(get_time()))
    #create an array
    assert x.shape == y.shape, 'The shapes dont match'
    dist_mx = np.zeros(shape=(x.shape[0], y.shape[0]))
    dist_mx[:] = np.inf
    i = 0
    j = 0
    for i in range(0, dist_mx.shape[0]):
        for j in range(0, dist_mx.shape[0]):
            dist_mx[i][j] = np.abs(x[i] - y[j])
    return dist_mx


def shortest_path(verbose, dist_arr, width=None):
    if verbose:
        print("calculating shortest distance; {}".format(get_time()))
    # should i start at the end or begining
    i, j, iteration = 0, 0, 0

    # when the final j is reached, the path is found
    if width is None:
        end_buffer = 0
    else:
        end_buffer = width

    last_i, last_j = dist_arr.shape[0]-1, dist_arr.shape[1]-1

    path = [] # keep each index, may not be necessary
    path_cost = [] # keep cost of each step
    while (i <= last_i-end_buffer and j != last_j) or (j <= last_j-end_buffer and i != last_i):
        if verbose:
            print('searching for shortest path index i: {};\tindex j: {}'.format(i,j))
        step, i, j = get_next_step(i, j, dist_arr, width) # [index, value] is this a problem? the distance matrix will be in memory now 3 times? or am i just passing by reference
        path.append([i, j])
        path_cost.append(step[1])
        iteration += 1

    return np.sum(path_cost), path, iteration

def get_next_step(i, j, dist_mx, w):
    # make an assertion regarding size of i and j?
    if w is None: # need to work on this!!!!!!!!
        w = dist_mx.shape[0] - max(i, j)
    print(w)

    # checking moves are within the warping width
    if i == 0 and j == 0:
        right, diagonal, down = dist_mx[i][j+1], dist_mx[i+1][j+1], dist_mx[i+1][j]
    elif (j - i) >= w:
        print('cant go right')
        diagonal, right, down = dist_mx[i+1][j+1], np.inf, dist_mx[i+1][j]
    elif (i - j) >= w:
        print('cant go down')
        diagonal, right, down = dist_mx[i+1][j+1], dist_mx[i][j+1], np.inf
    else:
        print('Went diagonal')
        diagonal, right, down = dist_mx[i+1][j+1], dist_mx[i][j+1], dist_mx[i+1][j]

    # find the cheapest move
    step_mx = [diagonal, right, down]
    next_step = [np.argmin(step_mx), min(step_mx)]  # [index, value]

    # which step was taken
    if next_step[0] == 0:  # diagonal step
        i += 1
        j += 1
    elif next_step[0] == 1:  # column increases, step right
        j += 1
    else: # step down
        i += 1

    return next_step, i, j

def main():
    verbose = True
    dtw_width = 2
    x = np.random.randint(0,10,10)
    y = np.random.randint(5,15,10)
    print(x)
    print(y)
    mx = create_dist_mx(False, x, y)
    print(mx)
    path_cost, path, num_iterations = shortest_path(verbose, mx, dtw_width)
    print(path_cost)
    print(path)

if __name__ == '__main__':
    main()



