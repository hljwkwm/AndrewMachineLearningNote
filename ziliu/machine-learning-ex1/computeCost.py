# coding:utf-8
import numpy as np


def compute_cost(X, y, theta):
    # 初始化
    m = y.size      # 数据集实例数量
    cost = 0        # 误差

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set the variable "cost" to the correct value.

    # 损失函数
    cost = np.sum((np.dot(X, theta) - y) ** 2) / (2 * m)

    return cost
