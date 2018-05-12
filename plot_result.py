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
    files = glob.glob(os.path.join(Settings.OUTPUT_DIR, '*.z'))
    for file in files:
        conditions = file.replace(".z", "").replace("cme_data", "cme").split('_')
        metric = conditions[1]
        approach = conditions[9]
        n_iter = conditions[-1]
        temp = conditions[5]
        res = joblib.load(file)
        df = pd.DataFrame(np.array(res[:-1]).T, columns=['n_samples', 'MSE', 'Preds'])
        df['approach'] = approach
        df['iteration'] = n_iter
        df['metric'] = metric
        df['temp'] = temp

        df_res = df_res.append(df)

    df_res['n_samples'] = df_res['n_samples'].astype(np.int32)
    g = sns.FacetGrid(df_res, col="temp", hue="approach", row="metric")
    g.map(sns.pointplot, "n_samples", "MSE")
    g.set(yscale="log")

    handlers = dict()
    for hue, color in zip(g.hue_names, g._colors):
        handlers[hue] = mpatches.Patch(color=color)

    g.add_legend(handlers)

    plt.savefig('realdata_result.png')