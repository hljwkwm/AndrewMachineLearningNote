import numpy as np
from computeCost import *


# 一维梯度下降
def gradient_descent(X, y, theta, alpha, num_iters):
	# 初始化
	m = y.size  # 数据集实例数量
	J_history = np.zeros(num_iters)  # 误差的记录

	for i in range(0, num_iters):
		# ===================== Your Code Here =====================
		# Instructions : Perform a single gradient step on the parameter vector theta
		#
		# Hint: X.shape = (97, 2), y.shape = (97, ), theta.shape = (2, )

		# 算出每个实例的误差
		error = np.dot(X, theta).flatten() - y
		# 迭代（注意，X第一列为1，第二列为具体的数据）
		# error[:, np.newaxis]是将error变成一个二维数组，即一个矩阵,shape为(m,1)
		theta -= (alpha / m) * np.sum(X * error[:, np.newaxis], 0)
		# 计算代价函数
		J_history[i] = compute_cost(X, y, theta)

	return theta, J_history


def gradient_descent_multi(X, y, theta, alpha, num_iters):
	# 初始化参数
	m = y.size
	J_history = np.zeros(num_iters)

	for i in range(0, num_iters):
		# ===================== Your Code Here =====================
		# Instructions : Perform a single gradient step on the parameter vector theta
		#
		# 发现和上面函数一样？
		error = np.dot(X, theta).flatten() - y
		theta -= (alpha / m) * np.sum(X * error[:, np.newaxis], 0)

		J_history[i] = compute_cost(X, y, theta)

	return theta, J_history
