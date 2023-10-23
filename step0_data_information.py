from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

root_dir = Path('.')
data_dir = root_dir / 'data'
figure_dir = root_dir / 'Figure'
figure_dir.mkdir(exist_ok=True, parents=True)

train_df_path = data_dir / 'train.csv'

train_df = pd.read_csv(train_df_path)
target = train_df['CI_HOUR'].to_numpy()

def draw_plot(df, plot_name) : 
    target = df['CI_HOUR'].to_numpy()
    # Plot histogram and boxplot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    sns.distplot(target, bins=100, kde=False, ax=ax)
    ax.set_title('Histogram of CI_HOUR')

    ax = axes[1]
    ax.boxplot(target)
    ax.set_title('Boxplot of CI_HOUR')
    plt.tight_layout()
    plt.savefig(figure_dir.joinpath(plot_name).with_suffix('.png'))
    plt.close()


# IDEA
## 1) Outlier가 굉장히 많음
## -> Outlier Detection + Regression이 필요할 것으로 보임

print(f"Number of data: {len(train_df)}")
print(f"50% quantile of CI_HOUR: {np.quantile(target, 0.5)}") # 8 hours
print(f"75% quantile of CI_HOUR: {np.quantile(target, 0.75)}") # 49 hours
print(f"95% quantile of CI_HOUR: {np.quantile(target, 0.95)}") # 276 hours

# 24시간의 quantile 확인
hour_24 = train_df[train_df['CI_HOUR'] <= 24].shape[0]
hour_24_q = train_df[train_df['CI_HOUR'] <= 24].shape[0] / train_df.shape[0]
print(f"24시간 이내의 데이터 개수: {hour_24} ({hour_24_q*100:.2f}%)") # 63.85%


under_24 = train_df.query('CI_HOUR <= 24')
over_24 = train_df.query('CI_HOUR > 24')

draw_plot(train_df, "TOTAL CI_HOUR")
draw_plot(under_24, "UNDER 24 CI_HOUR")
draw_plot(over_24, "OVER 24 CI_HOUR")

# NA 값 확인
print(train_df.isna().sum())

test_df = pd.read_csv(data_dir / 'test.csv')
print(test_df.isna().sum())