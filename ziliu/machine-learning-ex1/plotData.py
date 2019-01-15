import matplotlib.pyplot as plt


def plot_data(x, y):
	# ===================== Your Code Here =====================
	# Instructions : Plot the training data into a figure using the matplotlib.pyplot
	#                using the "plt.scatter" function. Set the axis labels using
	#                "plt.xlabel" and "plt.ylabel". Assume the population and revenue data
	#                have been passed in as the x and y.

	# Hint : You can use the 'marker' parameter in the "plt.scatter" function to change the marker type (e.g. "x", "o").
	#        Furthermore, you can change the color of markers with 'c' parameter.

	# ===================== 你的代码 =====================
	# Instructions : 使用matplotlib.pyplot中的"plt.scatter"将数据画在图标上
	#                使用"plt.xlabel"和"plt.ylabel"设置坐标系
	#                假设population和revenue data已作为x和y传入。

	# Hint : 你可以使用"plt.scatter"函数中的marker'参数来更改标记类型，如x、o等
	#        你也可以用参数'c'更改标记的颜色

	# 画分散图
	plt.scatter(x, y, c='r', marker="x")
	# 设置坐标系
	plt.xlabel('population')
	plt.ylabel('profit')
	# 显示图
	plt.show()
