from __future__ import print_function, division
from datetime import datetime
import pickle
import time
import numpy as np

class process_timer:
    '''
        A simple process timer.
        Initialize with a process name.
    '''
    def __init__(self, process_name, verbose=1):
        self.process_name = process_name
        self.verbose = verbose
    def __enter__(self):
        if self.verbose:
            print(self.process_name + " begin ...")
            self.begin_time = time.time()
    def __exit__(self, type, value, traceback):
        if self.verbose:
            end_time = time.time()
            print(self.process_name + " end ...")
            print('time lapsing {0} s \n'.format(end_time - self.begin_time))


def single_model_sub_to_csv(frame, cv_score, model_name=None, current_time=None):
    ''' Save submission to .csv with a time string as suffix

    :param cv_score:
    :param model_name:
    :param frame: submission data frame
    :param current_time:
    :return: N/A
    '''
    if current_time==None:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    score = (np.round(np.mean(cv_score), 6)).astype('str')
    str = current_time + '_' + model_name + '_' + score
    frame.to_csv('submission/single/sub{}.csv'.format(str), index=False)


def stacker_sub_to_csv(frame, current_time=None):
    ''' Save submission to .csv with a time string as suffix

    :param frame: submission data frame
    :param current_time:
    :return: N/A
    '''
    if current_time==None:
        frame.to_csv('submission/stacker/sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
    else:
        frame.to_csv('submission/stacker/sub{}.csv'.format(current_time), index=False)

def blender_sub_to_csv(frame, current_time=None):
    ''' Save submission to .csv with a time string as suffix

    :param frame: submission data frame
    :param current_time:
    :return: N/A
    '''
    if current_time==None:
        frame.to_csv('submission/blender/sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
    else:
        frame.to_csv('submission/blender/sub{}.csv'.format(current_time), index=False)

def model_to_dat(model, current_time=None):
    ''' Serialize the model to binary data

    :param model:
    :param current_time:
    :return: N/A
    '''
    if current_time==None:
        pickle.dump(model, open('model/mod{}.dat'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), 'wb'))
    else:
        pickle.dump(model, open('model/mod{}.dat'.format(current_time), 'wb'))

def dat_to_model(filename):
    ''' Retrieve a model from a .dat file

    :param filename: full file name including .dat
    :return: model
    '''
    model = pickle.load(open('model/' + filename, "rb"))
    return model

def model_metric_to_csv(frame, current_time=None):
    ''' Store evaluation metric values to csv file

    :param model:
    :param current_time:
    :return:
    '''
    if current_time==None:
        frame.to_csv('model_eval_metric/evals{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
    else:
        frame.to_csv('model_eval_metric/evals{}.csv'.format(current_time), index=False)

def save_submission_n_model(frame_sub, model):
    ''' Save all relevant data to files for future analysis

    :param frame_sub:
    :param model:
    :return:
    '''
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_to_dat(model, current_time)
    single_model_sub_to_csv(frame_sub, 0, None, current_time)
