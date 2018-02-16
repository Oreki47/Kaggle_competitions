
from lib.utilities import plot_heatmap
import numpy as np
import pandas as pd

import glob



tst_list = glob.glob('submission/blend/*.csv')
tst_series = pd.DataFrame()

for tst in tst_list:
	temp = pd.read_csv(tst, index_col='id')
	tst_series = pd.concat([tst_series, temp], axis=1)
tst_series.columns = [x[36:41] for x in tst_list]
plot_heatmap(tst_series.corr())