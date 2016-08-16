
# coding: utf-8

# > 在本篇中，我们将介绍Q宽客常用工具之一：函数插值。接着将函数插值应用于一个实际的金融建模场景中：波动率曲面构造。
# 
# > 通过本篇的学习您将学习到：
# 1. 如何在``scipy``中使用函数插值模块：``interpolate``；
# 2. 波动率曲面构造的原理；
# 3. 将``interpolate``运用于波动率曲面构造。

# 1. 如何使用``scipy``做函数插值
# -----------------------------
# ************

# 函数插值，即在离散数据的基础上补插连续函数，估算出函数在其他点处的近似值的方法。在``scipy``中，所有的与函数插值相关的功能都在``scipy.interpolate``模块中

# In[ ]:

from scipy import interpolate
dir(interpolate)[:5]


# 作为介绍性质的本篇，我们将只关注``interpolate.spline``的使用，即样条插值方法：
# 
# * ``xk``离散的自变量值，为序列
# * ``yk``对应``xk``的函数值，为与``xk``长度相同的序列
# * ``xnew``需要进行插值的自变量值序列
# * ``order``样条插值使用的函数基德阶数，为1时使用线性函数

# In[ ]:

print interpolate.spline.__doc__


# 1.1 三角函数（``np.sin``）插值
# ---------------
# 
# 一例胜千言！让我们这里用实际的一个示例，来说明如何在``scipy``中使用函数插值。这里的目标函数是三角函数：
# 
# $$f(x) = \mathrm{sin}(x)$$
# 
# 假设我们已经观测到的$f(x)$在离散点$x = (1,3,5,7,9,11,13)$的值：

# In[ ]:

import numpy as np
from matplotlib import pylab
import seaborn as sns
from CAL.PyCAL import *
font.set_size(20)
x = np.linspace(1.0, 13.0, 7)
y = np.sin(x)
pylab.figure(figsize = (12,6))
pylab.scatter(x,y, s = 85, marker='x', color = 'r')
pylab.title(u'$f(x)$离散点分布', fontproperties = font)


# 首先我们使用最简单的线性插值算法，这里面只要将``spline``的参数``order``设置为1即可：

# In[ ]:

xnew = np.linspace(1.0,13.0,500)
ynewLinear = interpolate.spline(x,y,xnew,order = 1)
ynewLinear[:5]


# 复杂一些的，也是``spline``函数默认的方法，即为样条插值，将``order``设置为3即可：

# In[ ]:

ynewCubicSpline = interpolate.spline(x,y,xnew,order = 3)
ynewCubicSpline[:5]


# 最后我们获得真实的$\mathrm{sin}(x)$的值：

# In[ ]:

ynewReal = np.sin(xnew)
ynewReal[:5]


# 让我们把所有的函数画到一起，看一下插值的效果。对于我们这个例子中的目标函数而言，由于本身目标函数是光滑函数，则越高阶的样条插值的方法，插值效果越好。

# In[ ]:

pylab.figure(figsize = (16,8))
pylab.plot(xnew,ynewReal)
pylab.plot(xnew,ynewLinear)
pylab.plot(xnew,ynewCubicSpline)
pylab.scatter(x,y, s = 160, marker='x', color = 'k')
pylab.legend([u'真实曲线', u'线性插值', u'样条插值', u'$f(x)$离散点'], prop = font)
pylab.title(u'$f(x)$不同插值方法拟合效果：线性插值 v.s 样条插值', fontproperties = font)


# 2. 函数插值应用 —— 期权波动率曲面构造
# ---------------------------------------
# ****************

# 市场上期权价格一般以隐含波动率的形式报出，一般来讲在市场交易时间，交易员可以看到类似的波动率矩阵（Volatilitie Matrix):

# In[ ]:

import pandas as pd
pd.options.display.float_format = '{:,>.2f}'.format
dates = [Date(2015,3,25), Date(2015,4,25), Date(2015,6,25), Date(2015,9,25)]
strikes = [2.2, 2.3, 2.4, 2.5, 2.6]
blackVolMatrix = np.array([[ 0.32562851,  0.29746885,  0.29260648,  0.27679993],
                  [ 0.28841840,  0.29196629,  0.27385023,  0.26511898],
                  [ 0.27659511,  0.27350773,  0.25887604,  0.25283775],
                  [ 0.26969754,  0.25565971,  0.25803327,  0.25407669],
                  [ 0.27773032,  0.24823248,  0.27340796,  0.24814975]])
table = pd.DataFrame(blackVolMatrix * 100, index = strikes, columns = dates, )
table.index.name = u'行权价'
table.columns.name = u'到期时间'
print u'2015年3月3日10时波动率矩阵'
table


# 交易员可以看到市场上离散值的信息，但是如果可以获得一些隐含的信息更好：例如，在2015年6月25日以及2015年9月25日之间，波动率的形状会是怎么样的？

# 2.1 方差曲面插值
# --------------------------

# 我们并不是直接在波动率上进行插值，而是在方差矩阵上面进行插值。方差和波动率的关系如下：
# 
# $$\mathrm{Var}(K,T) = \sigma(K,T)^2T$$
# 
# 所以下面我们将通过处理，获取方差矩阵（Variance Matrix):

# In[ ]:

evaluationDate = Date(2015,3,3)
ttm = np.array([(d - evaluationDate) / 365.0 for d in dates])
varianceMatrix = (blackVolMatrix**2) * ttm
varianceMatrix


# 这里的值``varianceMatrix``就是变换而得的方差矩阵。
# 
# 下面我们将在行权价方向以及时间方向同时进行线性插值，具体地，行权价方向：
# 
# $$\mathrm{Var}(K,t)= \frac{K_2 - K}{K_2 - K_1} \mathrm{Var}(K_1,t) + \frac{K - K_1}{K_2 - K_1} \mathrm{Var}(K_2,t)$$
# 
# 时间方向：
# 
# $$\mathrm{Var}(K) = \frac{t_2 - t}{t_2 - t_1} \mathrm{Var}(K,t_1) + \frac{t - t_1}{t_2 - t_1} \mathrm{Var}(K,t_2)$$
# 
# 这个过程在``scipy``中可以直接通过``interpolate``模块下``interp2d``来实现：
# 
# * ``ttm`` 时间方向离散点
# * ``strikes`` 行权价方向离散点
# * ``varianceMatrix`` 方差矩阵，列对应时间维度；行对应行权价维度
# * `` kind = 'linear'`` 指示插值以线性方式进行

# In[ ]:

interp = interpolate.interp2d(ttm, strikes, varianceMatrix, kind = 'linear')


# 返回的``interp``对象可以用于获取任意点上插值获取的方差值：

# In[ ]:

interp(ttm[0], strikes[0])


# 最后我们获取整个平面上所有点的方差值，再转换为波动率曲面。

# In[ ]:

sMeshes = np.linspace(strikes[0], strikes[-1], 400)
tMeshes = np.linspace(ttm[0], ttm[-1], 200)
interpolatedVarianceSurface = np.zeros((len(sMeshes), len(tMeshes)))
for i, s in enumerate(sMeshes):
    for j, t in enumerate(tMeshes):
        interpolatedVarianceSurface[i][j] = interp(t,s)
        
interpolatedVolatilitySurface = np.sqrt((interpolatedVarianceSurface / tMeshes))
print u'行权价方向网格数：', np.size(interpolatedVolatilitySurface, 0)
print u'到期时间方向网格数：', np.size(interpolatedVolatilitySurface, 1)


# 选取某一个到期时间上的波动率点，看一下插值的效果。这里我们选择到期时间最近的点：2015年3月25日：

# In[ ]:

pylab.figure(figsize = (16,8))
pylab.plot(sMeshes, interpolatedVolatilitySurface[:, 0])
pylab.scatter(x = strikes, y = blackVolMatrix[:,0], s = 160,marker = 'x', color = 'r')
pylab.legend([u'波动率（线性插值）', u'波动率（离散）'], prop = font)
pylab.title(u'到期时间为2015年3月25日期权波动率', fontproperties = font)


# 最终，我们把整个曲面的图像画出来看看：

# In[ ]:

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

maturityMesher, strikeMesher = np.meshgrid(tMeshes, sMeshes)
pylab.figure(figsize = (16,9))
ax = pylab.gca(projection = '3d')
surface = ax.plot_surface(strikeMesher, maturityMesher, interpolatedVolatilitySurface*100, cmap = cm.jet)
pylab.colorbar(surface,shrink=0.75)
pylab.title(u'2015年3月3日10时波动率曲面', fontproperties = font)
pylab.xlabel("strike")
pylab.ylabel("maturity")
ax.set_zlabel(r"volatility(%)")

