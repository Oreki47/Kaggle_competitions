import Tit_lib as tl
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import matplotlib.pyplot as plt


df = tl.load_file('train_rescaled.csv')
g = sns.pairplot(df)
plt.show()
