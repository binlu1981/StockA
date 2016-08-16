
# coding: utf-8

# ### Python数据处理的瑞士军刀：pandas

# ####第一篇：基本数据结构介绍

# ####一、Pandas介绍
# 
# 终于写到了作者最想介绍，同时也是Python在数据处理方面功能最为强大的扩展模块了。在处理实际的金融数据时，一个条数据通常包含了多种类型的数据，例如，股票的代码是字符串，收盘价是浮点型，而成交量是整型等。在C++中可以实现为一个给定结构体作为单元的容器，如向量（vector，C++中的特定数据结构）。在Python中，pandas包含了高级的数据结构Series和DataFrame，使得在Python中处理数据变得非常方便、快速和简单。
# 
# pandas不同的版本之间存在一些不兼容性，为此，我们需要清楚使用的是哪一个版本的pandas。现在我们就查看一下量化实验室的pandas版本：

# In[ ]:

import pandas as pd
pd.__version__


# pandas主要的两个数据结构是Series和DataFrame，随后两节将介绍如何由其他类型的数据结构得到这两种数据结构，或者自行创建这两种数据结构，我们先导入它们以及相关模块：

# In[ ]:

import numpy as np
from pandas import Series, DataFrame


# ####二、Pandas数据结构：Series
# 
# 从一般意义上来讲，Series可以简单地被认为是一维的数组。Series和一维数组最主要的区别在于Series类型具有索引（index），可以和另一个编程中常见的数据结构哈希（Hash）联系起来。

# #####2.1 创建Series
# 
# 创建一个Series的基本格式是s = Series(data, index=index, name=name)，以下给出几个创建Series的例子。首先我们从数组创建Series：

# In[ ]:

a = np.random.randn(5)
print "a is an array:"
print a
s = Series(a)
print "s is a Series:"
print s


# 可以在创建Series时添加index，并可使用Series.index查看具体的index。需要注意的一点是，当从数组创建Series时，若指定index，那么index长度要和data的长度一致：

# In[ ]:

s = Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
print s
s.index


# 创建Series的另一个可选项是name，可指定Series的名称，可用Series.name访问。在随后的DataFrame中，每一列的列名在该列被单独取出来时就成了Series的名称：

# In[ ]:

s = Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'], name='my_series')
print s
print s.name


# Series还可以从字典（dict）创建：

# In[ ]:

d = {'a': 0., 'b': 1, 'c': 2}
print "d is a dict:"
print d
s = Series(d)
print "s is a Series:"
print s


# 让我们来看看使用字典创建Series时指定index的情形（index长度不必和字典相同）：

# In[ ]:

Series(d, index=['b', 'c', 'd', 'a'])


# 我们可以观察到两点：一是字典创建的Series，数据将按index的顺序重新排列；二是index长度可以和字典长度不一致，如果多了的话，pandas将自动为多余的index分配NaN（not a number，pandas中数据缺失的标准记号)，当然index少的话就截取部分的字典内容。

# 如果数据就是一个单一的变量，如数字4，那么Series将重复这个变量：

# In[ ]:

Series(4., index=['a', 'b', 'c', 'd', 'e'])


# #####2.2 Series数据的访问
# 
# 访问Series数据可以和数组一样使用下标，也可以像字典一样使用索引，还可以使用一些条件过滤：

# In[ ]:

s = Series(np.random.randn(10),index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
s[0]


# In[ ]:

s[:2]


# In[ ]:

s[[2,0,4]]


# In[ ]:

s[['e', 'i']]


# In[ ]:

s[s > 0.5]


# In[ ]:

'e' in s


# ####三、Pandas数据结构：DataFrame
# 
# 在使用DataFrame之前，我们说明一下DataFrame的特性。DataFrame是将数个Series按列合并而成的二维数据结构，每一列单独取出来是一个Series，这和SQL数据库中取出的数据是很类似的。所以，按列对一个DataFrame进行处理更为方便，用户在编程时注意培养按列构建数据的思维。DataFrame的优势在于可以方便地处理不同类型的列，因此，就不要考虑如何对一个全是浮点数的DataFrame求逆之类的问题了，处理这种问题还是把数据存成NumPy的matrix类型比较便利一些。

# #####3.1 创建DataFrame
# 
# 首先来看如何从字典创建DataFrame。DataFrame是一个二维的数据结构，是多个Series的集合体。我们先创建一个值是Series的字典，并转换为DataFrame：

# In[ ]:

d = {'one': Series([1., 2., 3.], index=['a', 'b', 'c']), 'two': Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
df = DataFrame(d)
print df


# 可以指定所需的行和列，若字典中不含有对应的元素，则置为NaN：

# In[ ]:

df = DataFrame(d, index=['r', 'd', 'a'], columns=['two', 'three'])
print df


# 可以使用dataframe.index和dataframe.columns来查看DataFrame的行和列，dataframe.values则以数组的形式返回DataFrame的元素：

# In[ ]:

print "DataFrame index:"
print df.index
print "DataFrame columns:"
print df.columns
print "DataFrame values:"
print df.values


# DataFrame也可以从值是数组的字典创建，但是各个数组的长度需要相同：

# In[ ]:

d = {'one': [1., 2., 3., 4.], 'two': [4., 3., 2., 1.]}
df = DataFrame(d, index=['a', 'b', 'c', 'd'])
print df


# 值非数组时，没有这一限制，并且缺失值补成NaN：

# In[ ]:

d= [{'a': 1.6, 'b': 2}, {'a': 3, 'b': 6, 'c': 9}]
df = DataFrame(d)
print df


# 在实际处理数据时，有时需要创建一个空的DataFrame，可以这么做：

# In[ ]:

df = DataFrame()
print df


# 另一种创建DataFrame的方法十分有用，那就是使用concat函数基于Serie或者DataFrame创建一个DataFrame

# In[ ]:

a = Series(range(5))
b = Series(np.linspace(4, 20, 5))
df = pd.concat([a, b], axis=1)
print df


# 其中的axis=1表示按列进行合并，axis=0表示按行合并，并且，Series都处理成一列，所以这里如果选axis=0的话，将得到一个10×1的DataFrame。下面这个例子展示了如何按行合并DataFrame成一个大的DataFrame：

# In[ ]:

df = DataFrame()
index = ['alpha', 'beta', 'gamma', 'delta', 'eta']
for i in range(5):
    a = DataFrame([np.linspace(i, 5*i, 5)], index=[index[i]])
    df = pd.concat([df, a], axis=0)
print df


# #####3.2 DataFrame数据的访问
# 
# 首先，再次强调一下DataFrame是以列作为操作的基础的，全部操作都想象成先从DataFrame里取一列，再从这个Series取元素即可。可以用datafrae.column_name选取列，也可以使用dataframe[]操作选取列，我们可以马上发现前一种方法只能选取一列，而后一种方法可以选择多列。若DataFrame没有列名，[]可以使用非负整数，也就是“下标”选取列；若有列名，则必须使用列名选取，另外datafrae.column_name在没有列名的时候是无效的：

# In[ ]:

print df[1]
print type(df[1])
df.columns = ['a', 'b', 'c', 'd', 'e']
print df['b']
print type(df['b'])
print df.b
print type(df.b)
print df[['a', 'd']]
print type(df[['a', 'd']])


# 以上代码使用了dataframe.columns为DataFrame赋列名，并且我们看到单独取一列出来，其数据结构显示的是Series，取两列及两列以上的结果仍然是DataFrame。访问特定的元素可以如Series一样使用下标或者是索引:

# In[ ]:

print df['b'][2]
print df['b']['gamma']


# 若需要选取行，可以使用dataframe.iloc按下标选取，或者使用dataframe.loc按索引选取：

# In[ ]:

print df.iloc[1]
print df.loc['beta']


# 选取行还可以使用切片的方式或者是布尔类型的向量：

# In[ ]:

print "Selecting by slices:"
print df[1:3]
bool_vec = [True, False, True, True, False]
print "Selecting by boolean vector:"
print df[bool_vec]


# 行列组合起来选取数据：

# In[ ]:

print df[['b', 'd']].iloc[[1, 3]]
print df.iloc[[1, 3]][['b', 'd']]
print df[['b', 'd']].loc[['beta', 'delta']]
print df.loc[['beta', 'delta']][['b', 'd']]


# 如果不是需要访问特定行列，而只是某个特殊位置的元素的话，dataframe.at和dataframe.iat是最快的方式，它们分别用于使用索引和下标进行访问：

# In[ ]:

print df.iat[2, 3]
print df.at['gamma', 'd']


# dataframe.ix可以混合使用索引和下标进行访问，唯一需要注意的地方是行列内部需要一致，不可以同时使用索引和标签访问行或者列，不然的话，将会得到意外的结果：

# In[ ]:

print df.ix['gamma', 4]
print df.ix[['delta', 'gamma'], [1, 4]]
print df.ix[[1, 2], ['b', 'e']]
print "Unwanted result:"
print df.ix[['beta', 2], ['b', 'e']]
print df.ix[[1, 2], ['b', 4]]


# ####参考文献
# 
# 1. http://pandas.pydata.org/pandas-docs/version/0.14.1
