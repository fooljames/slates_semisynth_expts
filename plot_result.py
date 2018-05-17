from sklearn.externals import joblib
import Settings
import os
import pandas as pd
import glob
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if __name__ == "__main__":

    df_res = pd.DataFrame()
    size = 'outputMedium' # output, outputLarge, outputMedium
    files = glob.glob(os.path.join(os.path.join(Settings.DATA_DIR, size), '*.z'))
    for file in files:
        conditions = file.replace(".z", "").replace("cme_data", "cme").split('_')
        res = joblib.load(file)
        df = pd.DataFrame(np.array(res[:-1]).T, columns=['n_samples', 'MSE', 'Preds'])
        df['approach'] = conditions[9]
        df['iteration'] = conditions[-1]
        df['metric'] = conditions[1]
        df['temp'] = conditions[5]
        df['M'] = int(conditions[3])
        df['L'] = int(conditions[4])
        df['M-L'] = conditions[3] + '-' + conditions[4]
        df['logger'] = conditions[6]
        df['ranker'] = conditions[7]

        df_res = df_res.append(df)

    df_res['n_samples'] = df_res['n_samples'].astype(np.int32)
    df_res['log_rmse'] = np.log10(np.sqrt(df_res['MSE']))

    df_plot = df_res.query("temp == 'n1.0' and logger == 'ftree' and ranker == 'elasso' and approach != 'CME'")

    g = sns.FacetGrid(df_plot, hue="approach",
                      col="M-L", row="metric",
                      size=3, aspect=1.5)
    g.map(sns.pointplot, "n_samples", "MSE")
    g.set(yscale="log")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Null - Lasso(Url), Target - Tree(Body)", size=13)

    handlers = dict()
    for hue, color in zip(g.hue_names, g._colors):
        handlers[hue] = mpatches.Patch(color=color)

    g.add_legend(handlers, title='Estimator')

    plt.savefig('realdata_result_medium_ftree_elasso.png')