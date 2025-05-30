#分析五种深度之间的相关性  bingo
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 1. 数据加载与预处理
def load_and_preprocess(filepath):
    """加载数据并进行初步处理"""
    data = pd.read_excel(filepath, index_col=0, parse_dates=[0])

    # 选择需要分析的列
    cols = ['0-5cm', '0-10cm', '10-40cm', '40-100cm', '100-200cm']
    data = data[cols]

    # 简单处理缺失值（用前一个值填充）
    data = data.ffill()

    # 滑动平均平滑处理（窗口大小24小时）
    for col in cols:
        data[col] = data[col].rolling(window=24, min_periods=1).mean()

    return data


# 2. 基本统计分析与相关性计算
def basic_analysis(data):
    """计算基本统计量和相关系数"""
    print("\n基本统计量:")
    print(data.describe())

    print("\n相关系数矩阵:")
    corr_matrix = data.corr()
    print(corr_matrix)

    return corr_matrix


# 3. 可视化分析
def visualize_correlation(data, corr_matrix):
    """可视化相关性分析结果"""
    # 绘制各深度时间序列
    plt.figure(figsize=(12, 8))
    for col in data.columns:
        plt.plot(data.index, data[col], label=col)
    plt.title('各深度土壤湿度时间序列')
    plt.xlabel('时间')
    plt.ylabel('土壤湿度')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制相关系数热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                fmt=".2f", annot_kws={"size": 12})
    plt.title('各深度土壤湿度相关系数矩阵')
    plt.show()

    # 绘制散点图矩阵
    sns.pairplot(data, diag_kind='kde')
    plt.suptitle('各深度土壤湿度散点图矩阵', y=1.02)
    plt.show()


# 4. 时间序列特异性分析
def time_series_analysis(data):
    """时间序列特异性分析"""
    # 交叉相关性分析
    pairs = [('0-5cm', '0-10cm'), ('0-5cm', '40-100cm'), ('0-10cm', '100-200cm')]

    for col1, col2 in pairs:
        plt.figure(figsize=(10, 4))
        ccf_values = ccf(data[col1].dropna(), data[col2].dropna(), adjusted=False)[:48]  # 分析48小时滞后

        plt.stem(range(len(ccf_values)), ccf_values, basefmt=" ")
        plt.axhline(y=0, color='black', linestyle='-')
        plt.axhline(y=2 / np.sqrt(len(data)), color='red', linestyle='--')
        plt.axhline(y=-2 / np.sqrt(len(data)), color='red', linestyle='--')
        plt.title(f'{col1}和{col2}的交叉相关函数(CCF)')
        plt.xlabel('滞后(小时)')
        plt.ylabel('相关系数')
        plt.show()

    # Granger因果检验示例
    print("\nGranger因果检验示例(0-5cm和0-10cm):")
    gc_res = grangercausalitytests(data[['0-5cm', '0-10cm']].dropna(), maxlag=3, verbose=True)


# 5. 自相关和偏自相关分析
def acf_pacf_analysis(data):
    """各变量的自相关和偏自相关分析"""
    for col in data.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(data[col].dropna(), lags=48, ax=ax1)
        plot_pacf(data[col].dropna(), lags=48, ax=ax2)
        plt.suptitle(f'{col}的自相关(ACF)和偏自相关(PACF)')
        plt.tight_layout()
        plt.show()


# 主函数
def main():
    # 数据加载
    filepath = 'hour.xlsx'
    data = load_and_preprocess(filepath)

    # 基本分析与相关性计算
    corr_matrix = basic_analysis(data)

    # 可视化分析
    visualize_correlation(data, corr_matrix)

    # 时间序列特异性分析
    time_series_analysis(data)

    # 自相关和偏自相关分析
    acf_pacf_analysis(data)


if __name__ == "__main__":
    main()