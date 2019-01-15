import matplotlib.pyplot as plt


def plot_data(x, y):
	# ===================== Your Code Here =====================
	# Instructions : Plot the training data into a figure using the matplotlib.pyplot
	#                using the "plt.scatter" function. Set the axis labels using
	#                "plt.xlabel" and "plt.ylabel". Assume the population and revenue data
	#                have been passed in as the x and y.

	# Hint : You can use the 'marker' parameter in the "plt.scatter" function to change the marker type (e.g. "x", "o").
	#        Furthermore, you can change the color of markers with 'c' parameter.

	# ===================== ��Ĵ��� =====================
	# Instructions : ʹ��matplotlib.pyplot�е�"plt.scatter"�����ݻ���ͼ����
	#                ʹ��"plt.xlabel"��"plt.ylabel"��������ϵ
	#                ����population��revenue data����Ϊx��y���롣

	# Hint : �����ʹ��"plt.scatter"�����е�marker'���������ı�����ͣ���x��o��
	#        ��Ҳ�����ò���'c'���ı�ǵ���ɫ

	# ����ɢͼ
	plt.scatter(x, y, c='r', marker="x")
	# ��������ϵ
	plt.xlabel('population')
	plt.ylabel('profit')
	# ��ʾͼ
	plt.show()
