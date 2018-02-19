import pandas as pd
import numpy as np
import time
def auto_downcast(frame, column_print=False, verbose=False):
    ''' Infer data type of a dataframe object and downcast through inferring

        IN PROGRESS

        Parameters
        ----------
        frame: Pandas Dataframe, target dataframe
        column_print: Boolean, if information of each column should be printed

        Return
        ------

        Example
        -------
    '''
    str_time = time.time()
    start_mem_usg = frame.memory_usage().sum() / 1024 ** 2
    if verbose:
        print"Memory usage of target dataframe is: %.2f MB" % start_mem_usg
    nalist = []  # Keeps track of columns that have missing values filled in.
    for col in frame.columns:
        if frame[col].dtype != object:  # Not strings

            # Print current column type
            if column_print:
                print("******************************")
                print("Column: ", col)
                print("dtype before: ", frame[col].dtype)

            # make variables for Int, max and min
            isInt = False
            mx = frame[col].max()
            mn = frame[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(frame[col]).all():
                nalist.append(col)
                frame[col].fillna(mn - 1, inplace=True)

            # test if column can be converted to an integer
            asint = frame[col].fillna(mn - 1).astype(np.int64)
            result = (frame[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                isInt = True

            # Make Integer/unsigned Integer datatypes
            if isInt:
                if mn >= 0:
                    if mx < 255:
                        frame[col] = frame[col].astype(np.uint8)
                    elif mx < 65535:
                        frame[col] = frame[col].astype(np.uint16)
                    elif mx < 4294967295:
                        frame[col] = frame[col].astype(np.uint32)
                    else:
                        frame[col] = frame[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        frame[col] = frame[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        frame[col] = frame[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        frame[col] = frame[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        frame[col] = frame[col].astype(np.int64)

            # Make float as float32
            else:
                frame[col] = frame[col].astype(np.float32)

            # Print new column type
            if column_print:
                print("dtype after: ", frame[col].dtype)
                print("******************************")

        else:
            # Fill N/A with -1 for strings.
            frame[col].fillna(-1, inplace=True)

    # Print final result
    end_time = time.time()
    mem_usg = frame.memory_usage().sum() / 1024 ** 2
    if verbose:
        print "Memory usage after downcast: %.3f MB" % mem_usg
        print "This is %f %% of the initial size" % (mem_usg * 100/float(start_mem_usg))
        print "Time used: %f s\n" % (end_time - str_time)
    return frame, nalist