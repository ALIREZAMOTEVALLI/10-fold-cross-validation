import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.animation as animation
import webbrowser
import os

# -------------------- Config --------------------
N_SPLITS = 10
SAMPLE_ROWS = 60
np.random.seed(42)

plt.rcParams.update({
    'font.size': 20, 'axes.labelsize': 20, 'axes.titlesize': 20,
    'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16
})

HORIZON_COLORS = {'A': 'red', 'B': 'green', 'C': 'orange'}
REGIONS = ['all', '1', '2', '3', '4', '5']


# -------------------- Helpers --------------------
def mean_bias_error(y_true, y_pred):
    return np.mean(y_pred - y_true)


def make_synth_dataset(horizon, n=SAMPLE_ROWS):
    if horizon == 'A':
        mu_true, sigma_true, sigma_pred, bias = 10, 3.0, 2.0, -0.3
    elif horizon == 'B':
        mu_true, sigma_true, sigma_pred, bias = 12, 4.0, 3.0, 0.6
    else:  # 'C'
        mu_true, sigma_true, sigma_pred, bias = 15, 5.0, 4.0, -1.0
    clay = np.clip(np.random.normal(mu_true, sigma_true, n), 0, None)
    pred = clay + np.random.normal(bias, sigma_pred, n)
    return pd.DataFrame({'clay': clay, 'clay_1': pred})


def kfold_metrics(df, n_splits=N_SPLITS):
    y_true = df['clay'].values
    y_pred = df['clay_1'].values
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse, mbe = [], []
    for _, test_idx in kf.split(y_true):
        yt, yp = y_true[test_idx], y_pred[test_idx]
        rmse.append(np.sqrt(mean_squared_error(yt, yp)))
        mbe.append(mean_bias_error(yt, yp))
    return rmse, mbe


# -------------------- Build synthetic data --------------------
names = []
dataframes = []
for horizon in ['A', 'B', 'C']:
    for region in REGIONS:
        names.append((horizon, region))
        dataframes.append(make_synth_dataset(horizon))

# -------------------- Compute metrics --------------------
rmse_data, mbe_data, horizons_for_boxes, x_labels = [], [], [], []
for (hz, rg), df in zip(names, dataframes):
    r, m = kfold_metrics(df)
    rmse_data.append(r)
    mbe_data.append(m)
    horizons_for_boxes.append(hz)
    x_labels.append(rg)

# -------------------- Animation --------------------
fig, axs = plt.subplots(1, 2, figsize=(18, 7))


def update(num):
    axs[0].clear()
    axs[1].clear()

    box_rmse = axs[0].boxplot(rmse_data[:num + 1], patch_artist=True, showmeans=False)
    for patch, hz in zip(box_rmse['boxes'], horizons_for_boxes[:num + 1]):
        patch.set_facecolor(HORIZON_COLORS.get(hz, 'lightgray'))
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    axs[0].set_title('RMSE (10-fold CV)')
    axs[0].set_ylabel('RMSE (%)')
    axs[0].set_xlabel('Geo-region')
    axs[0].set_xticks(range(1, len(x_labels[:num + 1]) + 1))
    axs[0].set_xticklabels(x_labels[:num + 1], rotation=90)

    box_mbe = axs[1].boxplot(mbe_data[:num + 1], patch_artist=True, showmeans=False)
    for patch, hz in zip(box_mbe['boxes'], horizons_for_boxes[:num + 1]):
        patch.set_facecolor(HORIZON_COLORS.get(hz, 'lightgray'))
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    axs[1].set_title('MBE (10-fold CV)')
    axs[1].set_ylabel('MBE (%)')
    axs[1].set_xlabel('Geo-region')
    axs[1].set_xticks(range(1, len(x_labels[:num + 1]) + 1))
    axs[1].set_xticklabels(x_labels[:num + 1], rotation=90)

    for ax in axs:
        for s in ax.spines.values():
            s.set_color('black')
            s.set_linewidth(1.2)

    legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=c, label=h, markersize=10)
        for h, c in HORIZON_COLORS.items()
    ]
    axs[0].legend(handles=legend_handles, title='Horizon', loc='upper left')


ani = animation.FuncAnimation(fig, update, frames=len(names), interval=500, repeat=False)
ani.save('kfold_validation.gif', writer='pillow')

# -------------------- Save summary --------------------
summary_df = pd.DataFrame({
    'Horizon': [hz for hz, _ in names],
    'Geo-region': x_labels,
    'RMSE_mean(%)': [np.mean(r) for r in rmse_data],
    'RMSE_std(%)': [np.std(r, ddof=1) for r in rmse_data],
    'MBE_mean(%)': [np.mean(m) for m in mbe_data],
    'MBE_std(%)': [np.std(m, ddof=1) for m in mbe_data],
})
summary_df.to_excel('validation_summary_10fold_example.xlsx', index=False)

# -------------------- Automatically open GIF --------------------
gif_path = os.path.abspath('kfold_validation.gif')
webbrowser.open(f'file://{gif_path}')

print("✅ Saved: kfold_validation.gif, validation_summary_10fold_example.xlsx")
print("✅ GIF opened in default viewer")
