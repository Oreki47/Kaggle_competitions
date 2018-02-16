SEED = 177

xgb_params_test = {}
xgb_params_test['eta'] = 0.1
xgb_params_test['nrounds'] = 50
xgb_params_test['objective'] = 'reg:linear'
xgb_params_test['eval_metric'] = 'rmse'
xgb_params_test['seed'] = SEED
xgb_params_test['silent'] = True
xgb_params_test['verbose_eval'] = False

xgb_params_full = {}
xgb_params_full['eta'] = 0.02
xgb_params_full['nrounds'] = 2000
xgb_params_full['objective'] = 'reg:linear'
xgb_params_full['eval_metric'] = 'rmse'
xgb_params_full['seed'] = SEED
xgb_params_full['silent'] = True
xgb_params_full['verbose_eval'] = False
xgb_params_full['early_stopping_rounds'] = 100

xgb_pick_params = {'full': xgb_params_full,
                'test': xgb_params_test}