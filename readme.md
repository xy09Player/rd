# laisi挑战赛
初1，决2

### 1. 库依赖
 - jieba(0.39)
 - gensim(3.4.0)
 - pytorch(0.4.1)
 - rouge(0.3.0)
 - matplotlib(2.2.2)
 - 中文词向量：https://pan.baidu.com/s/14JP1gD7hcmsWdSpTvA3vKA
 - 英文词向量：http://nlp.stanford.edu/data/glove.840B.300d.zip

### 2. 运行
 - data文件夹中存放基础数据：原始数据、外部中英文词向量
 - 英文词向量不可直接使用，需要调用data_pre/deal_glove.py 做额外处理
 - data_gen文件夹中存放训练与测试数据：程序会自动判断是否存在相关数据，防止重复生成
 - 训练： 在train.py(34~40行)中取消相应模型config注释，直接运行即可。基础参数在config_base.py中设定，模型参数在相应config中设定
 - 测试：在test.py(590行)中修改is_ensemble参数，is_ensemble=True表示运行集成模型，反之则是单模型，集成模型参数在config_ensemble.py中设定




