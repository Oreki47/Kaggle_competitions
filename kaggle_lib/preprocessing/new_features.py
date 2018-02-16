from ..utils.KA_utils import process_tracker
import numpy as np
import pandas as pd

def add_groupby_features_n_vs_1(frame, group_columns_list, target_columns_list, methods_list, keep_only_stats=True, verbose=1):
    '''Create statistical columns, group by [N columns] and compute stats on [1 column]

       Parameters
       ----------
       frame: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       target_columns_list: list_like
          column you want to compute stats, need to be a list with only one element
       methods_list: list_like
          methods that you want to use, all methods that supported by groupby in Pandas
       keep_only_stats: boolean
          only keep stats or return both raw columns and stats
       verbose: int
          1 return tick_tock info 0 do not return any info
       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
    '''
    with process_tracker("add stats features", verbose):
        dicts = {"group_columns_list": group_columns_list , "target_columns_list": target_columns_list, "methods_list" :methods_list}

        for k, v in dicts.items():
            try:
                if type(v) == list:
                    pass
                else:
                    raise TypeError(k + " should be a list")
            except TypeError as e:
                print(e)
                raise

        grouped_name = ''.join(group_columns_list)
        target_name = ''.join(target_columns_list)
        combine_name = [[grouped_name] + [method_name] + [target_name] for method_name in methods_list]

        df_new = frame.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped[target_name].agg(methods_list).reset_index()
        the_stats.columns = [grouped_name] + \
                            ['_%s_%s_by_%s' % (grouped_name, method_name, target_name) \
                             for (grouped_name, method_name, target_name) in combine_name]
        if keep_only_stats:
            return the_stats
        else:
            df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')
        return df_new

def add_groupby_features_1_vs_n(frame, group_columns_list, agg_dict, keep_only_stats=True, verbose=1):
    '''Create statistical columns, group by [N columns] and compute stats on [N column]

       Parameters
       ----------
       frame: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       agg_dict: python dictionary
          Dictionay used to create stats variables
       keep_only_stats: boolean
          only keep stats or return both raw columns and stats
       verbose: int
          1 return tick_tock info 0 do not return any info
       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
       {real_column_name: {your_specified_new_column_name : method}}
       agg_dict = {'user_id':{'prod_tot_cnts':'count'},
                   'reordered':{'reorder_tot_cnts_of_this_prod':'sum'},
                   'user_buy_product_times': {'prod_order_once':lambda x: sum(x==1),
                                              'prod_order_more_than_once':lambda x: sum(x==2)}}
       ka_add_stats_features_1_vs_n(train, ['product_id'], agg_dict)
    '''
    with process_tracker("add stats features", verbose):
        try:
            if type(group_columns_list) == list:
                pass
            else:
                raise TypeError(k + "should be a list")
        except TypeError as e:
            print(e)
            raise

        df_new = frame.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped.agg(agg_dict)
        the_stats.columns = the_stats.columns.droplevel(0)
        the_stats.reset_index(inplace=True)
        if keep_only_stats:
            df_new = the_stats
        else:
            df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')

    return df_new

def ka_replace_hash(hashes, hash_id_table):
    '''Replace "hash in hashes" to "numeric index in hash_id_table"

       Parameter
       ---------
       hashes: pandas series
       hash_id_table: pandas series

       Return
       ------
       numpy array:
           replaced numeric number


       Example
       -------
       user_ids:
       0        d9dca3cb44bab12ba313eaa681f663eb
       1        560574a339f1b25e57b0221e486907ed

       detail.USER_ID_hash:
       0         d9dca3cb44bab12ba313eaa681f663eb
       1         560574a339f1b25e57b0221e486907ed

       replace_hash(detail.USER_ID_hash, user_ids)
    '''
    replace_table = pd.Series(hash_id_table.index, index=hash_id_table.values)
    return replace_table[hashes].values

def ka_add_hash_feature(df, category_columns_list):
    '''Create hash column unique in your specified columns

       Parameters
       ----------
       df: pandas dataframe
           Features matrix

       category_columns_list: list_like
           column names in a list

       Return
       ------
       new pandas dataframe with original columns and new added columns
    '''
    with process_tracker("add hash feature"):
        df_new = df.copy()
        if(len(category_columns_list) > 8):
            df_new['hash_' + category_columns_list[0] + '_' + category_columns_list[-1]] = df_new[category_columns_list].apply(lambda x: hash(tuple(x)),
                                                                                               axis=1)
        else:
            df_new['hash_' + ''.join(category_columns_list)] = df_new[category_columns_list].apply(lambda x: hash(tuple(x)),
                                                                                               axis=1)
    return df_new
