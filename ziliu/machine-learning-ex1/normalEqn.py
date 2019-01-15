# coding:utf-8
import numpy as np


def normal_eqn(X, y):
    theta = np.zeros((X.shape[1], 1))

    # ===================== Your Code Here =====================
    # Instructions : Complete the code to compute the closed form solution
    #                to linear regression and put the result in theta
    #

    # Xt为X的转置
    Xt = np.transpose(X)
    # 用正规方程求解
    theta = np.linalg.pinv(Xt.dot(X)).dot(Xt).dot(y)    # θ = (X‘X)⁻¹X'y
    return theta
