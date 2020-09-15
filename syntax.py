import random

# 定语从句语法
grammar = '''
目的 => 咨询  ， 购买 。

咨询 => 主语 咨询 商品
购买 => 主语  方式 数值 

主语 => 客户一 | 客户二 | 客户三 | 客户四 | 客户五

咨询 => 通过人工客服咨询 | 通过机器人客服咨询  
商品 => 玩具 | 电器 | 服装 | 食品 | 

方式 => 用现金方式支付了 | 用代金券方式支付了
数值 => 1元 | 1000元 | 5000元 | 100元 

'''

# 得到语法字典
def getGrammarDict(gram, linesplit = "\n", gramsplit = "=>"):
    #定义字典
    result = {}

    for line in gram.split(linesplit):
        # 去掉首尾空格后，如果为空则退出
        if not line.strip(): 
            continue
        expr, statement = line.split(gramsplit)
        result[expr.strip()] = [i.split() for i in statement.split("|")]
    #print(result)
    return result

# 生成句子
def generate(gramdict, target, isEng = False):
    if target not in gramdict: 
        return target
    find = random.choice(gramdict[target])
    #print(find)
    blank = ''
    # 如果是英文中间间隔为空格
    if isEng: 
        blank = ' '
    return blank.join(generate(gramdict, t, isEng) for t in find)

gramdict = getGrammarDict(grammar)
print(generate(gramdict,"目的"))
print(generate(gramdict,"目的", True))


