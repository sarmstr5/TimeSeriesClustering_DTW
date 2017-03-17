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


def dtw_dist(verbose, x, y, w, get_path=False):
    # pass
    # calculate distance matrix
    dtw_dist_mx = create_dist_mx(verbose, x, y, w)
    dist = dtw_dist_mx[-1,-1]
    if get_path:
        dist, path, iterations = shortest_path(verbose, dtw_dist_mx, w)
        return dist, path, iterations
    else:
        return dist, None, None

def calculate_width(verbose):
    pass

def find_initial_j(i, dtw_width):
    if i - dtw_width > 0:
        initial_j = i - dtw_width
        # print('initial j: {}, i: {}'.format(initial_j, i))
    else:
        initial_j = 0
    return initial_j

def create_dist_mx(verbose, x, y, dtw_width):
    # create an array
    assert x.shape == y.shape, 'The shapes dont match'

    # create an extra column and row for path traverseral
    # makes it so I do not have to make conditional checks for out of index errors
    dist_mx = np.zeros(shape=(x.shape[0], y.shape[0]))
    dist_mx[:] = np.inf

    # Create distance matrix
    for i in range(0, dist_mx.shape[0]):
        # save time by starting within the dtw width
        initial_j = find_initial_j(i, dtw_width)
        for j in range(initial_j, dist_mx.shape[1]):
            if (j - i) > dtw_width:
                break
            dist_mx[i][j] = np.abs(x[i] - y[j])
    path_mx = np.insert(dist_mx,0,np.inf,axis=0)
    path_mx = np.insert(path_mx,0,np.inf,axis=1)
    path_mx[0,0] = 0
    # print(path_mx)

    # go back through summing the smallest cost step to find the optimal solution
    for i in range(1, path_mx.shape[0]):
        # save time by starting within the dtw width
        initial_j = find_initial_j(i, dtw_width)
        for j in range(initial_j, path_mx.shape[1]):
            current_step = path_mx[i,j]
            # print('i:{}\tj:{}\tvalue:{}'.format(i,j,current_step))
            if (j - i) > dtw_width:
                # print('break')
                break
            if(i==1 and j ==1) or (current_step == np.inf):
                # print('continue, current step:{}'.format(current_step))
                continue
            step = min([path_mx[i-1][j-1], path_mx[i][j-1], path_mx[i-1][j]])
            # print(step)
            # print(path_mx)
            path_mx[i, j] += step
    # print(path_mx)
    return path_mx[1:,1:]

def shortest_path(verbose, dist_arr, width=None):
    # should i start at the end or begining
    print(dist_arr)
    i, j, iterations = dist_arr.shape[0]-1, dist_arr.shape[1]-1, 0
    path_arr = dist_arr.copy()

    # when the final j is reached, the path is found
    path = [] # keep each index, may not be necessary
    path_cost = [] # keep cost of each step
    while (i != 0) or (j != 0):
        step, i, j = get_next_step(i, j, path_arr)
        path.append([i, j])
        path_cost.append(step[1])
        iterations += 1
    return np.sum(path_cost), path, iterations

def get_next_step(i, j, dist_mx):
    # make an assertion regarding size of i and j?
    # if step is outside of specified width, value is np.inf
    if (i != 0) and (j != 0):
        diagonal, left, up = dist_mx[i-1][j-1], dist_mx[i][j-1], dist_mx[i-1][j]
    elif j == 0:
        diagonal, left, up = np.inf, np.inf, dist_mx[i-1][j]
    else:
        diagonal, left, up = np.inf, dist_mx[i][j-1], np.inf

    # find the cheapest move
    # diagonal, right, down steps are chosen in that order in place of a tie
    step_mx = [diagonal, left, up]
    next_step = [np.argmin(step_mx), min(step_mx)]  # [index, value]

    # which step was taken
    if next_step[0] == 0:  # diagonal step
        i -= 1
        j -= 1
    elif next_step[0] == 1:  # column increases, step right
        j -= 1
    else: # step down
        i -= 1
    return next_step

def main():
    verbose = True
    dtw_width = 2
    x = np.random.randint(0,10,10)
    y = np.random.randint(5,15,10)
    print(x)
    print(y)
    mx = create_dist_mx(False, x, y, dtw_width)
    print(mx[-1,-1])
    print(mx)
    # path_cost, path, num_iterations = shortest_path(verbose, mx, dtw_width)
    # print(path_cost)
    # print(path)

if __name__ == '__main__':
    main()



