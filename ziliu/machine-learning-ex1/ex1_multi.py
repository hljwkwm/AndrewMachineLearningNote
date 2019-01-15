import matplotlib.pyplot as plt
import numpy as np
from featureNormalize import *
from gradientDescent import *
from normalEqn import *

plt.ion()

# ===================== Part 1: 特征规范化 =====================
print('加载数据...')
data = np.loadtxt('ex1data2.txt', delimiter=',', dtype=np.int64)
X = data[:, 0:2]
y = data[:, 2]
m = y.size

# 打印部分数据
print('数据集前十组数据: ')
for i in range(0, 10):
    print('x = {}, y = {}'.format(X[i], y[i]))

input('程序已暂停，按“回车”或者其他键继续...')

# 特征缩放并设置为0均值
print('归一化特征...')
# 归一化特征
X, mu, sigma = feature_normalize(X)
X = np.c_[np.ones(m), X]  # 在X前添加一列1

# ===================== Part 2: 梯度下降 =====================

# ===================== Your Code Here =====================
# Instructions : We have provided you with the following starter
#                code that runs gradient descent with a particular
#                learning rate (alpha).
#
#                Your task is to first make sure that your functions -
#                computeCost and gradientDescent already work with
#                this starter code and support multiple variables.
#
#                After that, try running gradient descent with
#                different values of alpha and see which one gives
#                you the best result.
#
#                Finally, you should complete the code at the end
#                to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: At prediction, make sure you do the same feature normalization.
#

print('开始运行梯度下降...')

# Choose some alpha value
alpha = 0.03            # 学习速率
num_iters = 400         # 迭代次数
theta = np.zeros(3)     # 初始化θ
# 运行梯度下降
theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)

# 绘制J的收敛图
plt.figure()
plt.plot(np.arange(J_history.size), J_history)
plt.xlabel(u'迭代次数')
plt.ylabel('Cost J')
plt.show()

# 显示梯度下降的结果
print('用梯度下降法计算的 Theta : \n{}'.format(theta))

# ============= 估计 ==================
# 估计 1650 sq-ft, 3 br 的价格
# ===================== Your Code Here =====================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
predict = np.array([1650, 3])       # 创建X
predict = (predict - mu) / sigma    # 归一化
predict = np.r_[(1, predict)]       # 前X前加一列1
price = np.dot(predict, theta)      # 估计

print('预测 1650 sq-ft, 3 br 房子的价格(使用梯度下降) : {:0.3f}'.format(price))

input('程序已暂停，按“回车”或者其他键继续...')

# ===================== Part 3: 正规方程 =====================

print('使用正规方程求解 ...')

# ===================== Your Code Here =====================
# Instructions : The following code computes the closed form
#                solution for linear regression using the normal
#                equations. You should complete the code in
#                normalEqn.py
#
#                After doing so, you should complete this code
#                to predict the price of a 1650 sq-ft, 3 br house.
#

# 加载数据
data = np.loadtxt('ex1data2.txt', delimiter=',', dtype=np.int64)
X = data[:, 0:2]
y = data[:, 2]
m = y.size

# 将截距项添加到X，在X前面加一列1
X = np.c_[np.ones(m), X]

# 正规方程求theta
theta = normal_eqn(X, y)

# 显示正规方程的结果
print('用正规方程计算的 Theta 值: \n{}'.format(theta))

# ============= 估计 ==================
# 估计 1650 sq-ft, 3 br 的价格
# ===================== Your Code Here =====================
predict = np.array([1, 1650, 3])    # 构建X
price = np.dot(predict, theta)      # 估计

print('预测 1650 sq-ft, 3 br 房子的价格(使用正规方程) : {:0.3f}'.format(price))

input('ex1_multi结束. 按“回车”或者其他键退出...')
