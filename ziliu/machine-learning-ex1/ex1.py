# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from computeCost import *
from gradientDescent import *
from plotData import *

# ===================== Part 1: 画图 =====================
print('画出数据...')
# 读取txt数据
data = np.loadtxt('ex1data1.txt', delimiter=',', usecols=(0, 1))
# 数据中的第一列作为X，第二列作为Y
X = data[:, 0]
y = data[:, 1]
m = y.size

# 打开plt交互模式
plt.ion()
plt.figure(0)
plot_data(X, y)     # 画图函数，该函数在一个单一的文件中

input('程序已暂停，按“回车”或者其他键继续...')

# ===================== Part 2: Gradient descent 梯度下降法 =====================
print('运行梯度下降法...')


X = np.c_[np.ones(m), X]  # 在X左面添加一列1
theta = np.zeros(2)  # 初始化拟合参数

# 梯度下降法参数设定
iterations = 1500       # 迭代次数
alpha = 0.01            # 学习速录

# 计算并显示初始化代价函数
print('初始化代价函数: ' + str(compute_cost(X, y, theta)) + ' (这个值大概为32.07)')
# 开始迭代，并记录代价函数的值
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

print('通过梯度下降法计算的θ: ' + str(theta.reshape(2)))

# 绘制线性拟合
plt.figure(0)
line1, = plt.plot(X[:, 1], np.dot(X, theta), label='Linear Regression')
plt.legend(handles=[line1])
plt.show()

input('程序已暂停， 按“回车”或其他按键继续...')

# 预测人口规模为35,000和70,000的值
predict1 = np.dot(np.array([1, 3.5]), theta)
print('当人口 = 35,000, 预测的数字为 {:0.3f} (这个数值应该约为 4519.77)'.format(predict1*10000))
predict2 = np.dot(np.array([1, 7]), theta)
print('当人口 = 70,000, 预测的数字为 {:0.3f} (这个数值应该约为 45342.45)'.format(predict2*10000))

input('程序已暂停，按“回车”或者其他键继续...')

# ===================== Part 3: 可视化 J(theta0, theta1) =====================
print('可视化 J(theta0, theta1) ...')

# 创建-10到10和-1到4的等差数列
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# 生成网格点
xs, ys = np.meshgrid(theta0_vals, theta1_vals)
# 根据网格点的数量生成J
J_vals = np.zeros(xs.shape)

# 计算代价函数的值
for i in range(0, theta0_vals.size):
    for j in range(0, theta1_vals.size):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i][j] = compute_cost(X, y, t)

J_vals = np.transpose(J_vals)

# 画3D图
fig1 = plt.figure(1)
ax = fig1.gca(projection='3d')
ax.plot_surface(xs, ys, J_vals)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.show()

plt.figure(2)
lvls = np.logspace(-2, 3, 20)
# 画等高线图
plt.contour(xs, ys, J_vals, levels=lvls, norm=LogNorm())
# 在等高线图上加中心点
plt.plot(theta[0], theta[1], c='r', marker="x")
plt.show()

input('ex1已结束，按“回车”或者其他键退出...')
