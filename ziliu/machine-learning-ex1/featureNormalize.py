# coding:utf-8
import numpy as np


def feature_normalize(X):
    # You need to set these values correctly
    n = X.shape[1]          # 特征数量
    X_norm = X              # 归一化X值
    mu = np.zeros(n)        # μ值（均值）
    sigma = np.zeros(n)     # Σ值（特征缩放）

    # ===================== Your Code Here =====================
    # Instructions : First, for each feature dimension, compute the mean
    #                of the feature and subtract it from the dataset,
    #                storing the mean value in mu. Next, compute the
    #                standard deviation of each feature and divide
    #                each feature by its standard deviation, storing
    #                the standard deviation in sigma
    #
    #                Note that X is a 2D array where each column is a
    #                feature and each row is an example. You need
    #                to perform the normalization separately for
    #                each feature.
    #
    # Hint: You might find the 'np.mean' and 'np.std' functions useful.
    #       To get the same result as Octave 'std', use np.std(X, 0, ddof=1)
    #

    # 介绍：首先，对于每个特征维度，计算特征的平均值
    #      并从数据集中减去它，将平均值存储在mu中。
    #      接下来，计算每个特征的标准偏差，并将每个
    #      特征除以其标准偏差，将标准偏差存储在西格
    #      玛中。
    #      X是一个二维数组，每列是一个特征，每行是一
    #      个实例。你需要对每个特征都执行归一化。

    # 按列求均值
    mu = np.mean(X, 0)
    # 求标准差，0为按列求，ddof表示除数中N-ddof，ddof=1为无偏标准差
    sigma = np.std(X, 0, ddof=1)
    # 归一化特征
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma
