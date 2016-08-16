
# coding: utf-8

# ### Python数据处理的瑞士军刀：pandas

# ####第二篇：快速进阶

# 在上一篇中我们介绍了如何创建并访问pandas的Series和DataFrame类型的数据，本篇将介绍如何对pandas数据进行操作，掌握这些操作之后，基本可以处理大多数的数据了。首先，导入本篇中使用到的模块：

# In[ ]:

import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# 为了看数据方便一些，我们设置一下输出屏幕的宽度

# In[ ]:

pd.set_option('display.width', 200)


# ####一、数据创建的其他方式
# 
# 数据结构的创建不止是上篇中介绍的标准形式，本篇再介绍几种。例如，我们可以创建一个以日期为元素的Series：

# In[ ]:

dates = pd.date_range('20150101', periods=5)
print dates


# 将这个日期Series作为索引赋给一个DataFrame：

# In[ ]:

df = pd.DataFrame(np.random.randn(5, 4),index=dates,columns=list('ABCD'))
print df


# 只要是能转换成Series的对象，都可以用于创建DataFrame：

# In[ ]:

df2 = pd.DataFrame({ 'A' : 1., 'B': pd.Timestamp('20150214'), 'C': pd.Series(1.6,index=list(range(4)),dtype='float64'), 'D' : np.array([4] * 4, dtype='int64'), 'E' : 'hello pandas!' })
print df2


# ####二、数据的查看
# 
# 在多数情况下，数据并不由分析数据的人员生成，而是通过数据接口、外部文件或者其他方式获取。这里我们通过量化实验室的数据接口获取一份数据作为示例：

# In[ ]:

stock_list = ['000001.XSHE', '000002.XSHE', '000568.XSHE', '000625.XSHE', '000768.XSHE', '600028.XSHG', '600030.XSHG', '601111.XSHG', '601390.XSHG', '601998.XSHG']
raw_data = DataAPI.MktEqudGet(secID=stock_list, beginDate='20150101', endDate='20150131', pandas='1')
df = raw_data[['secID', 'tradeDate', 'secShortName', 'openPrice', 'highestPrice', 'lowestPrice', 'closePrice', 'turnoverVol']]


# 以上代码获取了2015年一月份全部的交易日内十支股票的日行情信息，首先我们来看一下数据的大小：

# In[ ]:

print df.shape


# 我们可以看到有200行，表示我们获取到了200条记录，每条记录有8个字段，现在预览一下数据，dataframe.head()和dataframe.tail()可以查看数据的头五行和尾五行，若需要改变行数，可在括号内指定：

# In[ ]:

print "Head of this DataFrame:"
print df.head()
print "Tail of this DataFrame:"
print df.tail(3)


# dataframe.describe()提供了DataFrame中纯数值数据的统计信息：

# In[ ]:

print df.describe()


# 对数据的排序将便利我们观察数据，DataFrame提供了两种形式的排序。一种是按行列排序，即按照索引（行名）或者列名进行排序，可调用dataframe.sort_index，指定axis=0表示按索引（行名）排序，axis=1表示按列名排序，并可指定升序或者降序：

# In[ ]:

print "Order by column names, descending:"
print df.sort_index(axis=1, ascending=False).head()


# 第二种排序是按值排序，可指定列名和排序方式，默认的是升序排序：

# In[ ]:

print "Order by column value, ascending:"
print df.sort(columns='tradeDate').head()
print "Order by multiple columns value:"
df = df.sort(columns=['tradeDate', 'secID'], ascending=[False, True])
print df.head()


# ####三、数据的访问和操作
# 
# #####3.1 再谈数据的访问
# 
# 上篇中已经介绍了使用loc、iloc、at、iat、ix以及[]访问DataFrame数据的几种方式，这里再介绍一种方法，使用":"来获取部行或者全部列：

# In[ ]:

print df.iloc[1:4][:]


# 我们可以扩展上篇介绍的使用布尔类型的向量获取数据的方法，可以很方便地过滤数据，例如，我们要选出收盘价在均值以上的数据：

# In[ ]:

print df[df.closePrice > df.closePrice.mean()].head()


# isin()函数可方便地过滤DataFrame中的数据：

# In[ ]:

print df[df['secID'].isin(['601628.XSHG', '000001.XSHE', '600030.XSHG'])].head()
print df.shape


# #####3.2 处理缺失数据
# 
# 在访问数据的基础上，我们可以更改数据，例如，修改某些元素为缺失值：

# In[ ]:

df['openPrice'][df['secID'] == '000001.XSHE'] = np.nan
df['highestPrice'][df['secID'] == '601111.XSHG'] = np.nan
df['lowestPrice'][df['secID'] == '601111.XSHG'] = np.nan
df['closePrice'][df['secID'] == '000002.XSHE'] = np.nan
df['turnoverVol'][df['secID'] == '601111.XSHG'] = np.nan
print df.head(10)


# 原始数据的中很可能存在一些数据的缺失，就如同现在处理的这个样例数据一样，处理缺失数据有多种方式。通常使用dataframe.dropna()，dataframe.dropna()可以按行丢弃带有nan的数据；若指定how='all'（默认是'any'），则只在整行全部是nan时丢弃数据；若指定thresh，则表示当某行数据非缺失列数超过指定数值时才保留；要指定根据某列丢弃可以通过subset完成。

# In[ ]:

print "Data size before filtering:"
print df.shape

print "Drop all rows that have any NaN values:"
print "Data size after filtering:"
print df.dropna().shape
print df.dropna().head(10)

print "Drop only if all columns are NaN:"
print "Data size after filtering:"
print df.dropna(how='all').shape
print df.dropna(how='all').head(10)

print "Drop rows who do not have at least six values that are not NaN"
print "Data size after filtering:"
print df.dropna(thresh=6).shape
print df.dropna(thresh=6).head(10)

print "Drop only if NaN in specific column:"
print "Data size after filtering:"
print df.dropna(subset=['closePrice']).shape
print df.dropna(subset=['closePrice']).head(10)


# 有数据缺失时也未必是全部丢弃，dataframe.fillna(value=value)可以指定填补缺失值的数值

# In[ ]:

print df.fillna(value=20150101).head()


# #####3.3 数据操作
# 
# Series和DataFrame的类函数提供了一些函数，如mean()、sum()等，指定0按列进行，指定1按行进行：

# In[ ]:

df = raw_data[['secID', 'tradeDate', 'secShortName', 'openPrice', 'highestPrice', 'lowestPrice', 'closePrice', 'turnoverVol']]
print df.mean(0)


# value_counts函数可以方便地统计频数：

# In[ ]:

print df['closePrice'].value_counts().head()


# 在panda中，Series可以调用map函数来对每个元素应用一个函数，DataFrame可以调用apply函数对每一列（行）应用一个函数，applymap对每个元素应用一个函数。这里面的函数可以是用户自定义的一个lambda函数，也可以是已有的其他函数。下例展示了将收盘价调整到[0, 1]区间：

# In[ ]:

print df[['closePrice']].apply(lambda x: (x - x.min()) / (x.max() - x.min())).head()


# 使用append可以在Series后添加元素，以及在DataFrame尾部添加一行：

# In[ ]:

dat1 = df[['secID', 'tradeDate', 'closePrice']].head()
dat2 = df[['secID', 'tradeDate', 'closePrice']].iloc[2]
print "Before appending:"
print dat1
dat = dat1.append(dat2, ignore_index=True)
print "After appending:"
print dat


# DataFrame可以像在SQL中一样进行合并，在上篇中，我们介绍了使用concat函数创建DataFrame，这就是一种合并的方式。另外一种方式使用merge函数，需要指定依照哪些列进行合并，下例展示了如何根据security ID和交易日合并数据：

# In[ ]:

dat1 = df[['secID', 'tradeDate', 'closePrice']]
dat2 = df[['secID', 'tradeDate', 'turnoverVol']]
dat = dat1.merge(dat2, on=['secID', 'tradeDate'])
print "The first DataFrame:"
print dat1.head()
print "The second DataFrame:"
print dat2.head()
print "Merged DataFrame:"
print dat.head()


# DataFrame另一个强大的函数是groupby，可以十分方便地对数据分组处理，我们对2015年一月内十支股票的开盘价，最高价，最低价，收盘价和成交量求平均值：

# In[ ]:

df_grp = df.groupby('secID')
grp_mean = df_grp.mean()
print grp_mean


# 如果希望取每只股票的最新数据，应该怎么操作呢？drop_duplicates可以实现这个功能，首先对数据按日期排序，再按security ID去重：

# In[ ]:

df2 = df.sort(columns=['secID', 'tradeDate'], ascending=[True, False])
print df2.drop_duplicates(subset='secID')


# 若想要保留最老的数据，可以在降序排列后取最后一个记录，通过指定take_last=True（默认值为False，取第一条记录）可以实现：

# In[ ]:

print df2.drop_duplicates(subset='secID', take_last=True)


# ####四、数据可视化
# 
# pandas数据直接可以绘图查看，下例中我们采用中国石化一月的收盘价进行绘图，其中set_index('tradeDate')['closePrice']表示将DataFrame的'tradeDate'这一列作为索引，将'closePrice'这一列作为Series的值，返回一个Series对象，随后调用plot函数绘图，更多的参数可以在matplotlib的文档中查看。

# In[ ]:

dat = df[df['secID'] == '600028.XSHG'].set_index('tradeDate')['closePrice']
dat.plot(title="Close Price of SINOPEC (600028) during Jan, 2015")


# ####参考文献
# 
# 1. http://pandas.pydata.org/pandas-docs/version/0.14.1 
