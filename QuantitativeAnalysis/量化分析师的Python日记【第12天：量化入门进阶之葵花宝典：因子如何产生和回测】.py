
# coding: utf-8

# ####  **0 预备知识**

# **预备知识包括：数学，计算机，投资学**
# 
# 数学方面至少包括微积分，线性代数，优化理论，概率统计基础，线性回归等知识点。数学出生最佳，一般理工科都基本满足要求，即使有所欠缺，花点时间也就自学补上了
# 
# 计算机主要有两点：一是会编程；二是会数据分析，相信在量化实验室看到截止今天日记的同学都已经满足了
# 
# 投资学方面只要通过大学的《投资学》课程就好，像William Sharpe等3人合著的《投资学》，要是能够通过CFA那就最好，知识面更广

# ####  **1 入门阶段**

# **Barra USE3 handbook**
# 
# Barra是量化投资技术提供商,是量化投资先驱。其经典的美国股票风险模型第3版（USE3）手册，详细介绍了股票市场多因子模型的理论框架和实证细节。手册共几十页，不太长，描述规范清晰，不陷入无意义的细节，非常适合于入门, [**点此下载**](http://datayes.oss-cn-hangzhou.aliyuncs.com/barra_handbook_US.pdf)

# ####  **2 系统学习阶段**

# **系统学习1：Quantitative Equity Portfolio Management（QEPM）， Ludwig Chincarini 偏学术风格。**
# 
# 偏学术界的作者撰写的关于量化股票组合投资的系统教程。尤其是前几章概述部分写得非常精彩、易懂、准确。把该领域的各个方面高屋建瓴地串讲了一遍。后面部分的章节似乎略有些学术了，但也值得一读。由于其较高的可读性，适于初学者学习。

# **系统学习2：Active Portfolio Management（APM）， Grinold & Kahn 偏业界风格。**
# 
# 业界先驱所著，作者均曾任Barra公司的研究总监。本书深度相对较深，描述也偏实践，介绍了许多深刻的真知。并且书中很多论述精彩而透彻。该书被奉为量化组合投资业界圣经。不过该书有些章节撰写得深度不一，初学者容易感到阅读起来有点困难。所以推荐：首次阅读不必纠结看不懂的细节，只要不影响后续阅读就跳过具体细节；有一定基础后，建议经常反复阅读本书。

# **系统学习3：Quantitative Equity Portfolio Management（QEPM）， Qian & Hua & Sorensen APM的补充**
# 
# 业界人士所著。针对性地对APM没有展开讲的一些topic做了很好的深入探讨。建议在APM之后阅读。该书风格比较数学，不过对数学专业背景的人并不太难。撰写文字也比较流畅。
# 
# 注：修行上述3本葵花宝典是否要割舍些什么？主要是与亲友坐在一起聊天喝茶的时光、一些睡觉的时间以及购书需要上千元钱（建议读英文原著）；好消息是，练成之后，不仅钱可以赚回来，空闲时间也会多起来。

# ####  **3 实践阶段**

# 券商卖方金工研究报告：多因子模型、选股策略、择时策略
# 
# 系统学习上面的材料之后，你已经有了分辨能力，这是看数量众多的券商卖方金工研究报告，就可以庖丁解牛，分辨真伪，总能筛选出优质信息积累下来了。
# 
# 值得总结的是数学、计算机、分析框架等工具都只是量化投资的形，优质投资想法才是灵魂。所以在修炼上述量化投资的基本功的同时，请不要忘记向有洞察力、有独立思考的其它派系的投资专家学习，无论他/她是价值投资、成长投资、涨停板敢死队、技术分析、主题投资、逆向投资、各类套利。将你自己想出的或者从别人那里习得的投资想法，用量化框架验证、改进、去伪存真，并最终上实盘创造价值。
# 
# 最推荐的入行过程：学习上述材料的同时，在通联量化实验室利用海量数据编程实现，理论付诸实践！

# #### **4 实战操作示例**

# 在关于pandas的前两篇介绍中，我们已经接触了不少关于Series和DataFrame的操作以及函数。本篇将以实际的例子来介绍pandas在处理实际金融数据时的应用。
# 
# 因子选股是股票投资中最常用的一种分析手段，利用量化计算的因子从成百上千的股票中进行快速筛选，帮助投资者从海量的数据中快速确定符合要求的目标，以下我们以量化因子计算过程的实例来展示如何利用pandas处理数据。

# 首先，我们依然是导入需要的一些外部模块：

# In[ ]:

import numpy as np
import pandas as pd
import datetime as dt
from pandas import Series, DataFrame, isnull
from datetime import timedelta, datetime
from CAL.PyCAL import *

pd.set_option('display.width', 200)


# 接着我们定义股票池和计算时所需要的时间区间参数。通常而言，计算某个因子是基于全A股的，这里作为示例，以HS300作为股票池。以计算市净率（PB）为例，我们取近一年的数据：

# In[ ]:

universe = set_universe('HS300')

today = Date.todaysDate()
start_date = (today - Period('1Y')).toDateTime().strftime('%Y%m%d')
end_date = today.toDateTime().strftime('%Y%m%d')
print 'start_date'
print start_date
print 'end_date'
print end_date


# 市净率是每股市价(Price)和每股净资产(Book Value)的比值，计算时通常使用总市值和归属于母公司所有者权益合计之比得到。前者通过访问股票日行情数据可以获得，后者在资产负债表上能够查到。在量化实验室中提供了访问股票日行情和资产负债表的API，可以获得相应数据。需要注意的一点是在获取财务报表数据时，因为只能指定一种类型的财报（季报，半年报，年报），需要做一个循环查询，并将获取到的DataFrame数据按垂直方向拼接，这里使用了concat函数：

# In[ ]:

market_capital = DataAPI.MktEqudGet(secID=universe, field=['secID', 'tradeDate', 'marketValue', 'negMarketValue'], beginDate=start_date, endDate=end_date, pandas='1')

equity = DataFrame()
for rpt_type in ['Q1', 'S1', 'Q3', 'A']:
    try:
        tmp = DataAPI.FdmtBSGet(secID=universe, field=['secID', 'endDate', 'publishDate', 'TEquityAttrP'], beginDate=start_date, publishDateEnd=end_date,  reportType=rpt_type)
    except:
        tmp = DataFrame()
    equity = pd.concat([equity, tmp], axis=0)

print 'Data of TEquityAttrP:'
print equity.head()
print 'Data of marketValue:'
print market_capital.head()


# 对于市值的数据，每个交易日均有提供，实际上我们多取了数据，我们只需要最新的市值数据即可。为此，我们将数据按股票代码和交易日进行排序，并按股票代码丢弃重复数据。以下代码表示按股票代码和交易日进行升序排序，并在丢弃重复值时，保留最后一个（默认是第一个）：

# In[ ]:

market_capital = market_capital.sort(columns=['secID', 'tradeDate'], ascending=[True, True])
market_capital = market_capital.drop_duplicates(subset='secID', take_last=True)


# 并非所有的数据都是完美的，有时候也会出现数据的缺失。我们在计算时无法处理缺失的数据，需要丢弃。下面这一行代码使用了isnull函数检查数据中总市值的缺失值，返回的是一个等长的逻辑Series，若数据缺失则为True。为尽可能多利用数据，我们考虑在总市值缺失的情况下，若流通市值有数值，则使用流通市值替换总市值，仅在两者皆缺失的情况下丢弃数据（虽然多数情况下是流通市值缺失，有总市值的数据，这一处理方式在其它使用到流通市值计算的情形中可以参考）：

# In[ ]:

market_capital['marketValue'][isnull(market_capital['marketValue'])] = market_capital['negMarketValue'][isnull(market_capital['marketValue'])]


# 以下代码使用drop函数舍去了流通市值这一列，使用dropna函数丢弃缺失值，并使用rename函数将列marketValue重命名为numerator：

# In[ ]:

market_capital = market_capital.drop('negMarketValue', axis=1)
numerator = market_capital.dropna()
numerator.rename(columns={'marketValue': 'numerator'}, inplace=True)


# 我们可以看一下处理好的分子：

# In[ ]:

print numerator


# 接下来处理分母数据。同样，为保留最新数据，对权益数据按股票代码升序，报表日期和发布日期按降序排列。随后丢弃缺失数据并按照股票代码去掉重复项，更改列名TEquityAttrP为denominator：

# In[ ]:

equity = equity.sort(columns=['secID', 'endDate', 'publishDate'], ascending=[True, False, False])
equity = equity.dropna()
equity = equity.drop_duplicates(cols='secID')
denominator = equity
denominator.rename(columns={"TEquityAttrP": "denominator"}, inplace=True)


# 处理好的分母：

# In[ ]:

print denominator


# 分子分母处理好之后，我们将两个DataFrame使用merge函数合并，使用参数how='inner'保留在两者中均存在的股票。

# In[ ]:

dat_info = numerator.merge(denominator, on='secID', how='inner')


# 作为比值，分母不可以为零，这里我们通过设置分母绝对值大于一个很小的数来过滤不符合要求的数据。随后直接通过DataFrame['Column_name']的复制添加一列PB：

# In[ ]:

dat_info = dat_info[abs(dat_info['denominator']) >= 1e-8]
dat_info['PB'] = dat_info['numerator'] / dat_info['denominator']


# 将股票代码和PB值两列取出，使用set_index设置索引，此时，DataFrame就变成了一个Series了：

# In[ ]:

pb_signal = dat_info[['secID', 'PB']]
pb_signal = pb_signal.set_index('secID')['PB']
print pb_signal


# 好了接下来我们把以上PB因子计算过程变成一个函数，使得它可以计算回测开始时间到结束时间的PB值，这样我们可以在通联的多因子信号分析工具RDP中方便的测试
# 
# 

# In[ ]:

def str2date(date_str):
    date_obj = dt.datetime(int(date_str[0:4]), int(date_str[4:6]), int(date_str[6:8]))
    return Date.fromDateTime(date_obj)

def signal_pb_calc(universe, current_date):
    today = str2date(current_date)
    start_date = (today - Period('1Y')).toDateTime().strftime('%Y%m%d')
    end_date = today.toDateTime().strftime('%Y%m%d')
    # dealing with the numerator
    market_capital = DataAPI.MktEqudGet(secID=universe, field=['secID', 'tradeDate', 'marketValue', 'negMarketValue', 'turnoverVol'], beginDate=start_date, endDate=end_date, pandas='1')
    market_capital = market_capital[market_capital['turnoverVol'] > 0]
    market_capital = market_capital.sort(columns=['secID', 'tradeDate'], ascending=[True, True])
    market_capital = market_capital.drop_duplicates(subset='secID', take_last=True)
    market_capital['marketValue'][isnull(market_capital['marketValue'])] = market_capital['negMarketValue'][isnull(market_capital['marketValue'])]
    market_capital = market_capital.drop('negMarketValue', axis=1)
    numerator = market_capital.dropna()
    numerator.rename(columns={'marketValue': 'numerator'}, inplace=True)
    # dealing with the denominator
    equity = DataFrame()
    for rpt_type in ['Q1', 'S1', 'Q3', 'A']:
        try:
            tmp = DataAPI.FdmtBSGet(secID=universe, field=['secID', 'endDate', 'publishDate', 'TEquityAttrP'], beginDate=start_date, publishDateEnd=end_date,  reportType=rpt_type)
        except:
            tmp = DataFrame()
        equity = pd.concat([equity, tmp], axis=0)

    equity = equity.sort(columns=['secID', 'endDate', 'publishDate'], ascending=[True, False, False])
    equity = equity.dropna()
    equity = equity.drop_duplicates(cols='secID')
    denominator = equity
    denominator.rename(columns={"TEquityAttrP": "denominator"}, inplace=True)
    # merge two dataframe and calculate price-to- book ratio
    dat_info = numerator.merge(denominator, on='secID', how='inner')
    dat_info = dat_info[abs(dat_info['denominator']) >= 1e-8]
    dat_info['PB'] = dat_info['numerator'] / dat_info['denominator']
    pb_signal = dat_info[['secID', 'PB']]
    pb_signal["secID"] = pb_signal["secID"].apply(lambda x:x[:6])
    return pb_signal


# 此代码完成的功能是：
# 
# >* 计算沪深300成分股在一段时间内的PB值作为信号
# 
# >* 把这些PB数据按照天存储为csv文件
# 
# >* 把csv文件打包成zip
# 
# 可以将这些文件下载到本地，解压到一个文件夹（比如PB_for_Mercury_DEMO），然后上传到RDP（[通联策略研究](https://gw.wmcloud.com/rdp//#/signalMgr)）中当做信号使用。

# In[ ]:

start = datetime(2015, 1, 1)
end = datetime(2015, 4, 23)

univ = set_universe('HS300')
cal = Calendar('China.SSE')

all_files = []
today = start
while((today - end).days < 0):
    today_CAL = Date.fromDateTime(today)
    if(cal.isBizDay(today_CAL)):
        today_str = today.strftime("%Y%m%d")
        print "Calculating PB values on " + today_str
        pb_value = signal_pb_calc(univ, today_str)
        file_name = today_str + '.csv'
        pb_value.to_csv(file_name, index=False, header=False)
        all_files.append(file_name)
    today = today + timedelta(days=1)
    
# exporting all *.csv files to PB.zip
zip_files("PB"+ "_" + start.strftime("%Y%m%d") + "_" + end.strftime("%Y%m%d"), all_files)

# delete all *.csv
delete_files(all_files)


# 第一步：解压点击‘上传新信号’，选择信号文件夹并为信号命名，然后，开始上传；
# 
# ![image](http://datayes.oss-cn-hangzhou.aliyuncs.com/RDP_1.JPG)
# 
# 第二步：选中上传的新信号，点击 ‘开始回测’；
# 
# ![image](http://datayes.oss-cn-hangzhou.aliyuncs.com/RDP_2.JPG)
# 
# 第三步：进行回测的各种配置；
# 
# ![image](http://datayes.oss-cn-hangzhou.aliyuncs.com/RDP_3.JPG)
# 
# 第四步：开始回测，回测完成后，点击报告的链接，查看回测结果；
# 
# ![image](http://datayes.oss-cn-hangzhou.aliyuncs.com/RDP_4.JPG)
# 
# 第五步：查看回测结果。
# 
# ![image](http://datayes.oss-cn-hangzhou.aliyuncs.com/RDP_5.JPG)
# 

# 以上研究过程演示了单个因子的产生和快速回测的过程，后面会介绍多因子的策略框架
# 
# 
# ![image](http://datayes.oss-cn-hangzhou.aliyuncs.com/RDP_6.jpg)
# 

# #### 参考文献
# 
# 1. http://pandas.pydata.org/pandas-docs/version/0.14.1 
# 2. http://zhuanlan.zhihu.com/scientific-invest/19892626
