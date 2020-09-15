# Part1 导入
from surprise import Dataset
from surprise import Reader
from surprise import SlopeOne,BaselineOnly,accuracy
from surprise import KNNBasic,NormalPredictor
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import SVD
import pandas as pd
import time 
start = time.time()

# Part2 数据读取
    ##数据读取,先定义一个读取器，以','分隔，第一行跳过
reader=Reader(line_format='user item rating timestamp',sep=',',skip_lines=1)
data=Dataset.load_from_file('./ratings.csv',reader=reader)
    ##build_full_trainset()方法可用于在整个训练集上训练
train_set=data.build_full_trainset()


# Part3 使用SlopeOne算法
algo = SlopeOne()
algo.fit(train_set)

    ## 对指定用户和商品进行评分预测 raw user id (as in the ratings file). They are **strings**!
uid = str(196) 
iid = str(302)
    ## 输出uid对iid的预测结果
print("SlopeOne results:")
pred = algo.predict(uid, iid, r_ui=4, verbose=True)
    ## 定义K折交叉验证迭代器
kf=KFold(n_splits=3)
for trainset,testset in kf.split(data):
    algo.fit(trainset)
    predictions=algo.test(testset)
    ##计算RMSE
    accuracy.rmse(predictions,verbose=True)

print("-"*118)
end = time.time()
print("SlopeOne running time:", end-start)

print("-"*118)
start = time.time()

# Part4 使用Baseline算法,并进行优化
    # ALS优化，用字典的形式表示，n_epoch 是迭代次数，reg_u 是uesr项的正则化系数默认值为15，reg_i 是item 项的正则化系数默认值为10(https://surprise.readthedocs.io/en/stable/prediction_algorithms.html?highlight=reg_u#baselines-estimates-configuration)
bsl_options = {'method': 'als','n_epochs': 5,'reg_u': 12,'reg_i': 5}
algo = BaselineOnly(bsl_options=bsl_options)
#algo = BaselineOnly()
#基于正太分布预测随机评分 
#algo = NormalPredictor()

# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    # 训练并预测
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RMSE
    accuracy.rmse(predictions, verbose=True)

# raw user id (as in the ratings file). They are **strings**!
uid = str(196)
iid = str(302)
# 输出uid对iid的预测结果
print("BaselineOnly results:")
pred = algo.predict(uid, iid, r_ui=4, verbose=True)

print("-"*118)
end = time.time()
print("Baseline running time:", end-start)


#----------------------------------------------------------------------------------------------------
#Theory

#Surprise中的选用常用算法：
    # SlopeOne -> 协同过滤算法 (focus today)
    # Baseline算法 -> 基于统计的基准预测线打分 (focus today), 可以使用ALS/SGD进行优化
    # 基于邻域的协同过滤
    # 矩阵分解：SVD，SVD++，PMF，NMF
    # 

# Slopeone用于物品更新不频繁，数量相对较稳定并且物品数目明显小于用户数的场景。依赖用户的用户行为日志和物品偏好的相关内容。

#步骤：
    # Step1，计算Item之间的评分差的均值，记为评分偏差（两个item都评分过的用户）
    # Step2，根据Item间的评分偏差和用户的历史评分，预测用户对未评分的item的评分
    # Step3，将预测评分排序，取topN对应的item推荐给用户

#优点：
    #1.算法简单，易于实现，执行效率高；
    #2.可以发现用户潜在的兴趣爱好；
#缺点：
    #1. 依赖用户行为，存在冷启动问题和稀疏性问题。


#Baselineonly算法
#基准算法包含两个主要的算法NormalPredictor和BaselineOnly。
#BaselineOnly 是基于统计的基准预测线打分，思想是设立基线，并引入user的偏差以及item的偏差：
    #μ为所有用户对电影评分的均值
    #bui：待求的基线模型中用户u给物品i打分的预估值
    #bu：user偏差（如果用户比较苛刻，打分都相对偏低， 则bu<0；反之，bu>0）；
    #bi为item偏差，反映商品受欢迎程度

#Baselines can be estimated in two different ways:
    #-Using Stochastic Gradient Descent (SGD).
    #-Using Alternating Least Squares (ALS).

#使用ALS进行优化
    #Step1，固定bu，优化bi
    #Step2，固定bi，优化bu
#ALS参数:
    #reg_i：物品的正则化参数，默认为10。
    #reg_u：用户的正则化参数，默认为15 。
    #n_epochs：迭代次数，默认为10

