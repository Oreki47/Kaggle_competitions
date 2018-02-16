import sys, os
import pandas as pd

sys.path.append("../")
from general.ClfWrappers import XgbWrapper
from general.utilities import sub_to_csv
from features.f3 import prepare_data
from cv.cv_04 import cross_validate
from Models.ModelParams import xgb_pick_params

# Initialize
SEED = 177
TARGET = 'visitors'
MODEL_NAME = str(os.path.basename(sys.argv[0]).split('.')[0])
print("Overfiting process initiating with " + MODEL_NAME + "...")
full_data, ntrain, ntest, FEATURES, CAT_FEATS = prepare_data()

# Get parameters
try:
    exe_type = str(sys.argv[1])
    opt_type = int(sys.argv[2])
except:
    exe_type = 'test'
    opt_type = 0
opt_path = '../tuning/' + "_".join(MODEL_NAME.split('_')[:1]) + ".csv"
xgb_turn_params = dict(pd.read_csv(opt_type).iloc[opt_type, :])

xgb_params = {**xgb_pick_params[exe_type], **xgb_turn_params}

# Define model and get results
xgb_clf = XgbWrapper(seed=SEED, params=xgb_params)
results = cross_validate(
    full_data=full_data,
    clf=xgb_clf,
    seed=SEED,
    ntrain=ntrain,
    ntest=ntest,
    features=FEATURES,
    target=TARGET,
    nfolds=4,
)
sub, v06, v33, oof_score = results

# Save submission to file
sub_to_csv(sub, v06, v33, oof_score[0], oof_score[1], MODEL_NAME)
