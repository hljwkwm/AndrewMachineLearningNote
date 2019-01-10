# NumPy学习笔记

## 导入

```python
import numpy
```

## 基本操作

### 创建一个矩阵

```python
# numpy.array()可以用list或者list of list作为输入
# 当输入一个list，我们可以得到一个一维向量。
vector = numpy.array([5, 10, 15, 20])
# 当输入为一个list of list，将会得到一个矩阵
matrix = numpy.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
```

### 查看向量或者矩阵的维数或元素个数
```python
vector = numpy.array([1, 2, 3, 4])
print(vector.shape)

matrix = numpy.array([[5, 10, 15], [20, 25, 30]])
print(matrix.shape)

'''输出的结果为
(4,)
(2, 3)
'''
```

### 数据类型

在Numpy中，一个ndarray里的数据类型必须是一样的，如果不一样，系统会自动转换为相同的数据类型。

```python
# ndarray中每个值的数据类型都需要相同
# 在读取数据或将列表转换为数组时，NumPy会自动找出合适的数据类型。
# 您可以使用dtype属性检查NumPy数组的数据类型。
number1 = numpy.array([1, 2, 3, 4])
print(number1.dtype)
number2 = numpy.array([1, 2, 3, 4.])
print(number2.dtype)

'''
输出结果：
int32
float64
'''
```
注意：当NumPy无法将值转换为float或integer等数值数据类型时，它使用一个特殊的nan值代表不是一个数字。

### 数据读取

```python
# 读取一个数据
matrix = numpy.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
n1 = matrix[0, 1]
n2 = matrix[2, 2]
print(n1)
print(n2)
'''
输出结果：
10
45
'''
# 读取一组数据
vector = numpy.array([5, 10, 15, 20])
print(vector[0:3])
print('--------------------------')
matrix = numpy.array([
                    [5, 10, 15], 
                    [20, 25, 30],
                    [35, 40, 45]
                 ])
print(matrix[:,1])
print('--------------------------')
matrix = numpy.array([
                    [5, 10, 15], 
                    [20, 25, 30],
                    [35, 40, 45]
                 ])
print(matrix[:,0:2])
print('--------------------------')
matrix = numpy.array([
                    [5, 10, 15], 
                    [20, 25, 30],
                    [35, 40, 45]
                 ])
print(matrix[1:3,0:2])
'''
输出结果：
[ 5 10 15]
--------------------------
[10 25 40]
--------------------------
[[ 5 10]
 [20 25]
 [35 40]]
--------------------------
[[20 25]
 [35 40]]
'''
```

## 常用操作

### 比较和判断

```python
# 矩阵中每个元素都会和==后面的数字进行比较
# 如果相等，则返回True，如果不相等，则返回False
vector = numpy.array([5, 10, 15, 20])
print(vector == 10)
print('----------')
matrix = numpy.array([
                    [5, 10, 15], 
                    [20, 25, 30],
                    [35, 40, 45]
                 ])
print(matrix == 25)
'''
输出结果：
[False  True False False]
----------
[[False False False]
 [False  True False]
 [False False False]]

'''
```

利用一个矩阵和一个数字比较返回的新的bool类型的矩阵，可以作为该矩阵的索引，根据bool矩阵的真假，返回新的矩阵。
```python
vector = numpy.array([5, 10, 15, 20])
equal_to_ten = (vector == 10)
print(equal_to_ten)
print(vector[equal_to_ten])
'''
输出结果：
[False  True False False]
[10]

'''
```

读取第二列含有25的行
```python
matrix = numpy.array([
                [5, 10, 15], 
                [20, 25, 30],
                [35, 40, 45]
             ])
second_column_25 = (matrix[:,1] == 25)
print(second_column_25)
print(matrix[second_column_25, :])
'''
输出结果：
[False  True False]
[[20 25 30]]
'''
```

我们还可以与多个条件进行比较。

```python
vector = numpy.array([5, 10, 15, 20])
equal_to_ten_and_five = (vector >= 5) & (vector <= 15)
print(equal_to_ten_and_five)
print('----------')
vector = numpy.array([5, 10, 15, 20])
equal_to_ten_or_five = (vector == 10) | (vector == 5)
print(equal_to_ten_or_five)
print('----------')
vector = numpy.array([5, 10, 15, 20])
equal_to_ten_or_five = (vector == 10) | (vector == 5)
vector[equal_to_ten_or_five] = 50
print(vector)
print('----------')
matrix = numpy.array([
            [5, 10, 15], 
            [20, 25, 30],
            [35, 40, 45]
         ])
second_column_25 = matrix[:,1] == 25
print(second_column_25)
matrix[second_column_25, 1] = 10
print(matrix)
'''
输出结果：
[ True  True  True False]
----------
[ True  True False False]
----------
[50 50 15 20]
----------
[False  True False]
[[ 5 10 15]
 [20 10 30]
 [35 40 45]]
'''
```

### 类型转换

我们可以使用`ndarray.astype()`进行数据转换

```python
vector = numpy.array(["1", "2", "3"])
print vector.dtype
print vector
vector = vector.astype(float)
print vector.dtype
print vector
'''
输出结果：
|S1
['1' '2' '3']
float64
[ 1.  2.  3.]
'''
```

### 最大值和最小值

最大值和最小值可以使用`ndarray.max()`和`ndarray.min()`这两个函数。

```python
vector = numpy.array([5, 10, 15, 20])
max_num = vector.max()
min_num = vector.min()
print(max_num)
print(min_num)
'''
输出结果：
20
5
'''
```

### 求和

求和可以使用`ndarray.sum()`函数。
```python
vector = numpy.array([5, 10, 15, 20])
vector.sum()
'''
输出结果：
50
'''
```

`axis`指示我们在哪个维度上进行操作，当`axis=1`时，表示是对每行进行操作，`axis=0`表示对每列进行操作。
```python
matrix = numpy.array([
                [5, 10, 15], 
                [20, 25, 30],
                [35, 40, 45]
             ])
print(matrix.sum(axis=1))
print(matrix.sum(axis=0))
'''
输出结果：
[ 30  75 120]
[60 75 90
'''
```

E的n次幂以及对n开方：

```python
a = np.floor(10*np.random.random((2,2)))
'''
输出结果：
[[ 5.  6.]
 [ 1.  5.]]
'''
```

向下取整：

```python
B = np.arange(3)
print(B)
print(np.exp(B))
print(np.sqrt(B))
'''
输出结果：
[0 1 2]
[1.         2.71828183 7.3890561 ]
[0.         1.         1.41421356]

'''
```


## 常用变换

### 更改行列数目

更改行列数目可以使用`reshape()`函数。

```python
a = np.arange(15).reshape(3, 5)
print(a)
# 系统自动分配列
a = np.arange(15).reshape(3, -1)
'''
输出结果：
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
 [[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
'''
```

矩阵拉成一个向量：

```python
a = np.floor(10*np.random.random((3,4)))
print(a.ravel())
'''
输出结果：
[8. 9. 4. 8. 6. 8. 1. 6. 1. 6. 7. 0.]
'''
```

查看行列数目：

```python
a = np.arange(15).reshape(3, 5)
print(a.shape)
'''
结果：
(3, 5)
'''
```

查看维度：

```python
a = np.arange(15).reshape(3, 5)
print(a.ndim)
'''
结果：
2
'''
```

查看ndarray数据类型：

```python
a = np.arange(15).reshape(3, 5)
print(a.dtype.name)
'''
结果：
int32
'''
```

查看数组共有多少元素

```python
a = np.arange(15).reshape(3, 5)
print(a.size)
'''
结果：
15
'''
```

### 初始化

零矩阵：

```python
np.zeros ((3,4)) 
'''
结果：
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
'''
```

全1矩阵：

```python
np.ones( (2,3,4), dtype=np.int32 )
'''
结果：
array([[[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]],

       [[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]]])
'''
```

序列：

```python
np.arange( 10, 30, 5 )
'''
结果：
aarray([10, 15, 20, 25])
'''

np.arange( 0, 2, 0.3 )
'''
结果：
array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])
'''

np.linspace( 0, 99, 100 )
'''
结果：
array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
       13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
       26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38.,
       39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51.,
       52., 53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64.,
       65., 66., 67., 68., 69., 70., 71., 72., 73., 74., 75., 76., 77.,
       78., 79., 80., 81., 82., 83., 84., 85., 86., 87., 88., 89., 90.,
       91., 92., 93., 94., 95., 96., 97., 98., 99.])
'''
```

随机数：

```python
np.random.random((2,3))
'''
结果：
array([[ 0.40130659,  0.45452825,  0.79776512],
       [ 0.63220592,  0.74591134,  0.64130737]])
'''
```

### 转置

```python
a = np.floor(10*np.random.random((3,4)))
print(a.T)
'''
结果：
[[4. 7. 2.]
 [2. 1. 0.]
 [2. 0. 5.]
 [0. 0. 0.]]

'''
```

### 拼接

```python
a = np.floor(10*np.random.random((2,2)))
b = np.floor(10*np.random.random((2,2)))
print(a)
print('-----')
print(b)
print('-----')
# 水平拼接
print(np.hstack((a,b)))
print('-----')
# 垂直拼接
print(np.vstack((a,b)))
'''
结果：
[[3. 0.]
 [5. 9.]]
-----
[[8. 3.]
 [3. 5.]]
-----
[[3. 0. 8. 3.]
 [5. 9. 3. 5.]]
-----
[[3. 0.]
 [5. 9.]
 [8. 3.]
 [3. 5.]]
'''
```

### 分割

```python
a = np.floor(10*np.random.random((2,12)))
print(a)
print(np.hsplit(a,3))
print(np.hsplit(a,(3,4)))
a = np.floor(10*np.random.random((12,2)))
print(a)
print(np.vsplit(a,3))
'''
结果：
[[1. 3. 8. 4. 5. 8. 8. 8. 8. 8. 2. 2.]
 [2. 3. 8. 9. 8. 7. 3. 4. 0. 4. 3. 4.]]
[array([[1., 3., 8., 4.],
       [2., 3., 8., 9.]]), 
array([[5., 8., 8., 8.],
       [8., 7., 3., 4.]]), 
array([[8., 8., 2., 2.],
       [0., 4., 3., 4.]])]
[array([[1., 3., 8.],
       [2., 3., 8.]]), 
array([[4.],
       [9.]]), 
array([[5., 8., 8., 8., 8., 8., 2., 2.],
       [8., 7., 3., 4., 0., 4., 3., 4.]])]
[[5. 3.]
 [3. 3.]
 [1. 0.]
 [5. 3.]
 [6. 6.]
 [0. 9.]
 [9. 5.]
 [6. 4.]
 [1. 6.]
 [4. 4.]
 [9. 4.]
 [4. 2.]]
[array([[5., 3.],
       [3., 3.],
       [1., 0.],
       [5., 3.]]), 
array([[6., 6.],
       [0., 9.],
       [9., 5.],
       [6., 4.]]), 
array([[1., 6.],
       [4., 4.],
       [9., 4.],
       [4., 2.]])]
'''
```

### 复制

ndarray是引用类型，不可以直接复制。
```python
#Simple assignments make no copy of array objects or of their data.
a = np.arange(12)
b = a
# a and b are two names for the same ndarray object
print(b is a)
b.shape = (3,4)
print(a.shape)
print(id(a))
print(id(b))
'''
输出结果：
True
(3, 4)
2239129219120
2239129219120
'''
```

浅拷贝：
```python
#The view method creates a new array object that looks at the same data.
#a和c的结构不同，但是数据是共用的
c = a.view()
print(c is a)
c.shape = 2,6
print(a.shape)
c[0,4] = 1234
print(a)
print(id(a))
print(id(c))
'''
输出结果：
False
(3, 4)
[[   0    1    2    3]
 [1234    5    6    7]
 [   8    9   10   11]]
2239129219120
2239129219920
'''
```

深拷贝：

```python
#The copy method makes a complete copy of the array and its data.
d = a.copy() 
print(d is a)
d[0,0] = 9999
print(d)
print(a)
'''
输出结果：
False
[[9999    1    2    3]
 [1234    5    6    7]
 [   8    9   10   11]]
[[   0    1    2    3]
 [1234    5    6    7]
 [   8    9   10   11]]
'''
```

### 扩展

```python
a = np.arange(0, 40, 10)
b = np.tile(a, (3, 5)) 
print(b)
'''
输出结果：
[[ 0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30]
 [ 0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30]
 [ 0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30]]
'''
```



## 矩阵运算

加、减、乘方：
```python
a = np.array( [20,30,40,50] )
b = np.arange( 4 )
#print a 
#print b
#b
# shape相同的情况下，ndarray是对应元素加减
c = a-b
print(c)
print('----------')
# 如果ndarray加减一个数，则为每个元素对应加减这个数
d = a - 1
print(d)
print('----------')
# 乘方同理
print(b**2)
'''
结果：
[20 29 38 47]
----------
[19 29 39 49]
----------
[0 1 4 9]
'''
```

乘：
```python
A = np.array( [[1,1],
               [0,1]] )
B = np.array( [[2,0],
               [3,4]] )
print(A)
print(B)
print('----------')
# 对应元素乘
print(A*B)
print('----------')
# 矩阵相乘，两种方法
print(A.dot(B))
print(np.dot(A, B) )
'''
结果：
[[1 1]
 [0 1]]
[[2 0]
 [3 4]]
----------
[[2 0]
 [0 4]]
----------
[[5 4]
 [3 4]]
[[5 4]
 [3 4]]
'''
```

## 应用

### 取矩阵中的最大值

```python
import numpy as np
data = np.sin(np.arange(20)).reshape(5,4)
print(data)
# 取每列的最大值的索引
ind = data.argmax(axis=0)
print(ind)
# 取每列的最大值
data_max = data[ind, range(data.shape[1])]
print(data_max)
'''
输出结果：
[[ 0.          0.84147098  0.90929743  0.14112001]
 [-0.7568025  -0.95892427 -0.2794155   0.6569866 ]
 [ 0.98935825  0.41211849 -0.54402111 -0.99999021]
 [-0.53657292  0.42016704  0.99060736  0.65028784]
 [-0.28790332 -0.96139749 -0.75098725  0.14987721]]
[2 0 3 1]
[0.98935825 0.84147098 0.99060736 0.6569866 ]
'''
```

### 排序

```python
a = np.array([[4, 3, 5], [1, 2, 1]])
print(a)
print('--------------')
# 按行的顺序从小到大排列
b = np.sort(a, axis=1)
print(b)
a.sort(axis=1)
print(a)
print('--------------')
# 从小到大的索引值
a = np.array([4, 3, 1, 2])
j = np.argsort(a)
print(j)
# 按照从小到大的顺序打印
print(a[j])
'''
输出结果：
[[4 3 5]
 [1 2 1]]
--------------
[[3 4 5]
 [1 1 2]]
[[3 4 5]
 [1 1 2]]
--------------
[2 3 1 0]
[1 2 3 4]
'''
```

## 文件操作

**文件读取**

```python
# 第一个参数为文件的路径
# delimiter为分隔符类型
# dtype为读取数据后将数据转换为什么类型
world_alcohol = numpy.genfromtxt("world_alcohol.txt", delimiter=",", dtype=str)
```
## 其他

**帮助**

```python
print(help(numpy.genfromtxt))
```