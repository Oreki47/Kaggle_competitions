from matplotlib.pyplot import savefig
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint
import seaborn as sns


class process_timer:
    ''' A simple process timer.
        Initialize with a process name.

    '''
    def __init__(self, process_name, verbose=1):
        self.process_name = process_name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(self.process_name + " begin ...")
            self.begin_time = time.time()

    def __exit__(self, type_char, value, traceback):
        if self.verbose:
            end_time = time.time()
            print(self.process_name + " end ...")
            print('time lapsing {0} s \n'.format(end_time - self.begin_time))


def sub_to_csv(y_tst, y_trn, trn_gini, model_name, current_time):
    ''' Save submission with timestamp and score
        also save y_train_cv for blending

    '''
    score = (np.round(trn_gini, 6)).astype('str')
    str_val = current_time + '_' + model_name + '_' + score
    y_tst.to_csv('submission/sub{}.csv'.format(str_val), index=False)
    pd.DataFrame(y_trn).to_csv('train/train{}.csv'.format(str_val), index=False, header=False)

def sub_to_csv_blend(y_tst, trn_gini, current_time):
    ''' Save submission with timestamp and score
        from a simple blender

    '''
    score = (np.round(trn_gini, 6)).astype('str')
    str_val = current_time + '_' + 'blend01' + '_' + score
    y_tst.to_csv('submission/sub{}.csv'.format(str_val), index=False)


def plot_importance(fscore, cv_score, model_name=None, current_time=None):
    y_pos = np.arange(fscore.shape[0])
    mean_f = fscore.mean(axis=1).sort_values(ascending=False)
    plt.subplots(figsize=(20, 12))
    plt.barh(y_pos, mean_f)
    plt.yticks(y_pos, mean_f.index)
    if current_time is None:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    score = (np.round(np.mean(cv_score), 6)).astype('str')
    file_name = current_time + '_' + model_name + '_' + score
    fscore.to_csv('figures/fscore{}.csv'.format(file_name))
    savefig('figures/plot{}.pdf'.format(file_name), dpi=800)

def print_model_params(params_dict):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(params_dict)

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.pyplot import savefig
    from datetime import datetime


def plot_heatmap(corr, a=None):
    if a == None:
        a = [x for x in range(corr.shape[0])]

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr.iloc[a, a], dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(16, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr.iloc[a, a], mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    plt.show()