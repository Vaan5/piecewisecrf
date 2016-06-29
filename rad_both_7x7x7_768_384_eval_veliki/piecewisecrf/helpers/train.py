import time
from datetime import datetime

import tensorflow as tf


def get_variable_map():
    '''
    Returns dict of tensorflow variables (variable_name: variable)
    '''
    var_list = tf.all_variables()
    var_map = {}
    for var in var_list:
        var_map[var.name] = var
    return var_map


def get_time_string():
    '''
    Returns current time in day_month_HH-MM-SS/ format
    '''
    time = datetime.now()
    name = (str(time.day) + '_' + str(time.month) + '_%02d' % time.hour +
            '-%02d' % time.minute + '-%02d' % time.second + '/')
    return name


def get_time():
    '''
    Returns current time in HH:MM:SS format
    '''
    time = datetime.now()
    return '%02d' % time.hour + ':%02d' % time.minute + ':%02d' % time.second


def get_expired_time(start_time):
    '''

    Returns expired time in HH:MM:SS format calculated relative to start_time

    Parameters
    ----------
    start_time : int
        Starting point in time


    '''
    curr_time = time.time()
    delta = curr_time - start_time
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta
    return '%02d' % hour + ':%02d' % minute + ':%02d' % seconds
