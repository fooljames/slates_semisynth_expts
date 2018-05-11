from sklearn.externals import joblib
import Settings
import os
import pandas as pd
import glob
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    df_res = pd.DataFrame()
    files = glob.glob(os.path.join(Settings.OUTPUT_DIR , '*.z'))
    for file in files:
        conditions = file.replace(".z","").split("_")
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

    df_res_plot = df_res[df_res['metric'] == 'ERR']
    ax = sns.pointplot(x="n_samples", y="MSE", hue="approach", data=df_res_plot)
    ax.set_yscale('log')
    ax.set_ylabel("Mean Square Error (log scale)")

