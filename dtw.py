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
    # pass
    # calculate distance matrix
    dtw_dist_mx = create_dist_mx(verbose, x, y, w)
    # calculate shortest path
    dist, path, iterations = shortest_path(verbose, dtw_dist_mx, w)
    return dist, path, iterations

def calculate_width(verbose):
    pass

def find_initial_j(i, dtw_width):
    if i - dtw_width > 0:
        initial_j = i - dtw_width
    else:
        initial_j = 0
    return initial_j

def create_dist_mx(verbose, x, y, dtw_width):
    # create an array
    assert x.shape == y.shape, 'The shapes dont match'
    dist_mx = np.zeros(shape=(x.shape[0], y.shape[0]))
    dist_mx[:] = np.inf
    i = 0
    j = 0
    for i in range(0, dist_mx.shape[0]):
        # save time by starting within the dtw width
        initial_j = find_initial_j(i, dtw_width)
        for j in range(initial_j, dist_mx.shape[1]):
            if (j - i) > dtw_width:
                break
            dist_mx[i][j] = np.abs(x[i] - y[j])
    return dist_mx

def shortest_path(verbose, dist_arr, width=None):
    # should i start at the end or begining
    i, j, iterations = 0, 0, 0

    # when the final j is reached, the path is found
    last_i, last_j = dist_arr.shape[0]-1, dist_arr.shape[1]-1

    path = [] # keep each index, may not be necessary
    path_cost = [] # keep cost of each step
    while (i != last_i) or (j != last_j):
        step, i, j = get_next_step(i, j, dist_arr) # [index, value] is this a problem? the distance matrix will be in memory now 3 times? or am i just passing by reference
        path.append([i, j])
        path_cost.append(step[1])
        iterations += 1
    return np.sum(path_cost), path, iterations

def get_next_step(i, j, dist_mx):
    # make an assertion regarding size of i and j?
    # if step is outside of specified width, value is np.inf
    if (i != dist_mx.shape[0]-1) and (j != dist_mx.shape[0]-1):
        diagonal, right, down = dist_mx[i+1][j+1], dist_mx[i][j+1], dist_mx[i+1][j]
    elif j == dist_mx.shape[1]-1:
        diagonal, right, down = np.inf, np.inf, dist_mx[i+1][j]
    else:
        diagonal, right, down = np.inf, dist_mx[i][j+1], np.inf

    # find the cheapest move
    # diagonal, right, down steps are chosen in that order in place of a tie
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
    mx = create_dist_mx(False, x, y, dtw_width)
    print(mx)
    path_cost, path, num_iterations = shortest_path(verbose, mx, dtw_width)
    print(path_cost)
    print(path)

if __name__ == '__main__':
    main()



