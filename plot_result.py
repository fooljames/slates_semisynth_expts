from sklearn.externals import joblib
import Settings
import os
import pandas as pd
import glob
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.lines as mlines

if __name__ == "__main__":

    df_res = pd.DataFrame()
    size = 'outputStochastic copy'  # output, outputLarge, outputMedium, outputStochastic
    files = glob.glob(os.path.join(os.path.join(Settings.DATA_DIR, size), '*.z'))

    for file in files:
        conditions = file.replace(".z", "").replace("cme_data", "cme").split('_')
        res = joblib.load(file)
        df = pd.DataFrame(np.array(res[:-1]).T, columns=['n_samples', 'MSE', 'Preds'])
        df['estimator'] = conditions[9]
        df['iteration'] = conditions[-1]
        df['metric'] = conditions[1]
        df['temp'] = conditions[5]
        df['m'] = int(conditions[3])
        df['l'] = int(conditions[4])
        df['ml'] = conditions[3] + ',' + conditions[4]
        df['logger'] = conditions[6]
        df['ranker'] = conditions[7]

        df_res = df_res.append(df)

    df_res['n_samples'] = df_res['n_samples'].astype(np.int32)
    df_res['log_rmse'] = np.log10(np.sqrt(df_res['MSE']))

    df_plot = df_res.query("temp == 'n1.0' and logger == 'ftree' "
                           "and metric == 'ERR' and ranker == 'elasso'")

    sns.set_style("ticks")

    sns.set(font_scale=1.5)

    hue_kws = {'linestyles': ["-", "--", "-.", ":", ':', '-'],
               'markers': ['x', 'v', 'o', '^', 's', 'd'],
               'capsize': [0.05] * 6}

    subplot_kws = {'markeredgewidth': [] * 5, 'markeredgecolor': ['black'] * 5}

    g = sns.FacetGrid(df_plot, hue="estimator", hue_order=['CME-A', 'DM-tree', 'DR', 'PI-SN', 'IPS-SN', 'OnPolicy'],
                      col="ml",
                      size=4.0, aspect=1.2, legend_out=True,
                      hue_kws=hue_kws, ylim=(5e-7, 1.2))

    g.map(sns.pointplot, "n_samples", "MSE", scale=1.25)

    g.set(yscale="log")
    g.set_ylabels("Mean Square Error")
    g.set_xlabels("Number of observations")

    plt.tight_layout()

    g.axes[0][0].set_title("M=100, K=10")
    g.axes[0][1].set_title("M=10, K=5")

    handlers = dict()

    g.hue_names = ['CME', 'Direct', 'DR', 'Slate', 'wIPS', 'OnPolicy']
    for hue, color, marker in zip(g.hue_names, g._colors, hue_kws['markers']):
        if marker == 'x':
            marker = "X"
        handlers[hue] = mlines.Line2D([], [], color=color, marker=marker, markersize=13)

    g.add_legend(handlers, title='Estimator')
    xticks = [1000, 2500, 6300, 16000, 40000, 100000]
    g.axes[0][0].xaxis.set_major_locator(plt.FixedLocator(range(len(xticks))))
    g.axes[0][1].xaxis.set_major_formatter(ticker.FixedFormatter(xticks))

    g.fig.get_children()[-1].set_bbox_to_anchor((1.01, 0.5, 0, 0))

    plt.subplots_adjust(right=0.9)

    plt.savefig('realdata_result_stochastic.pdf', format='pdf')
