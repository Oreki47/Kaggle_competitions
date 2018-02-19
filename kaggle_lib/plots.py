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

def plot_importance(fscore, cv_score, current_time=None, model_name=None):
    y_pos = np.arange(fscore.shape[0])
    mean_f = fscore.mean(axis=1).sort_values(ascending=False)
    f, ax = plt.subplots(figsize=(20, 12))
    plt.barh(y_pos, mean_f)
    plt.yticks(y_pos, mean_f.index)
    if current_time==None:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    score = (np.round(np.mean(cv_score), 6)).astype('str')
    file_name = current_time + '_' + model_name + '_' + score
    savefig('figures/sub{}.pdf'.format(file_name), dpi=800)

def plot_missing_vals(frame):
	cols = frame.columns
	missing_counts = frame.isnull().sum(axis=0)
	missing_df = pd.DataFrame([cols, missing_counts]).transpose()
	missing_df.columns = ['feature', 'counts']
	missing_df = missing_df.sort_values('counts')

	fig, ax = plt.subplots()
	fig.set_size_inches(20, 12)
	y_pos = np.arange(len(missing_df.counts))

	ax.barh(y_pos, missing_df.counts)
	ax.set_yticks(np.arange(len(missing_df.counts)))
	ax.set_yticklabels(missing_df.feature)
	for i, v in enumerate(missing_df.counts):
	    ax.text(v, i - 0.35, str(v), color='black', fontweight='bold')