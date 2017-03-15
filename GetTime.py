from datetime import datetime as dt


class GetTime(object):

    def __init__(self):
        self. initial_time = time()


def time():
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
