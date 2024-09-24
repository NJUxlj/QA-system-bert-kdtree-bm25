# QA-system-bert-kdtree-bm25
一个基于FAQ知识库的问答系统，应用了多种文本匹配算法, 包括：
1. 基于bert和qwen的表示型文本匹配模型。
   1. Bert+TextRCNN
   2. Qwen-0.5B
   3. Bert+Qwen-0.5B
2. BM25匹配算法。
3. KD树文本搜索算法


### This project is not finished yet


## Question-Answer System 的结构

### FAQ问题知识库
- taget: 标准问
- questions：和标准问含义相同的问题集合 （相似问）


```python
# faq knowledge base 是一个 json line file
# 每一行是一个json object，形如：
{"questions": ["宽带坏了", "网断了", "有线宽带断了", "宽带不能用了找谁", "网速太慢了", "网慢的都不能用了", "宽带出现了问题找人帮我修一下", "显示宽带连接那一直是个感叹号", "电信宽带有毛病能有人来修吗"], "target": "有限宽带障碍报修"}

```


### 文本匹配算法
- 用到的基于统计的文本匹配算法
  - BM25 (精排)
  - KD-Tree 
- 用到的文本相似度预测模型
  - Bert
  - Bert+Text-RCNN+2xMLP
  - Qwen-0.5B + MLP
  - Bert + Qwen0.5B + MLP
- 用到的基于深度学习的文本匹配系统
  - 表示型文本匹配系统
  - 交互型文本匹配系统

---

### 召回阶段 （粗排）


#### KV召回
1. 使用TF-IDF来提取每个标准问的关键词， 将faq知识库中的每一行转为 `{关键词 : 标准问+相似问}` 字典键值对元素.
2. 最后构建成一个KV索引，其中键为 关键词，值为 标准问+相似问。
3. 当接收到查询请求时，系统根据查询中的关键词在KV索引中查找对应的键。返回与这些键相关联的值，即候选数据项。
4. 这些候选数据项将被传递到后续的排序阶段进行进一步处理。

### 排序阶段 （精排）
- 将用户问匹配到的所有问题按照相似度进行排序
- 如果之前用了BM25，那就已经排过序了，这里就不排了。
- 其他模型视情况在最后加上一层BM25



### BM25算法简介



### KD-Tree算法简介


### 表示型文本匹配模型
- the input sentences A ann B first go through their berts, and form 2 sentence representations, and then connnect 2 representations to 1 bug vector, and send it into a classifer to classify into label 1:match, 0: not match.
![image](https://github.com/user-attachments/assets/9c0a2be1-6074-42ab-a5d8-f82e41f837b6)

![image](https://github.com/user-attachments/assets/8d7252ed-f152-4090-bd9a-62a3de0040f2)


### 交互型文本匹配模型
- the sentence A and B will first go through the their word-embeddings, after getting A and B's sentence vectors, we will connnet them into 1 piece and send it into a bert model, and then go through the classifier.
![image](https://github.com/user-attachments/assets/4b11e102-fe4a-413b-b15f-979f6b4d21d7)


---

## 项目结构
- similarity_function.py:
  -  jaccard相似度算法
  -  编辑距离相似度算法  

## Requirements
```shell
# 由于我的环境中有几百个包，各位直接install requirements会比较麻烦，所以我直接把需要的包列在这里
pickle
jieba
gensim
torch
transformers
gradio
datasets
numpy
```

## 模型下载+配置
1. 先把`bert-base-chinese`模型下载下来，放到你的本地目录下
2. 再把`qwen-0.5B`模型下载下来，放到你的本地目录下
3. 修改`config.py`的相应字段

```python
Config = {
    "model_path": "model_output",
    "schema_path": "../data/schema.json",
    "data_path":"../data/data.json",
    "train_data_path": "../data/train.json",
    "valid_data_path": "../data/valid.json",
    "vocab_path":"../chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 200,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "labels_count": 2,
    
    "bert_config":{
        "model_path":r"D:\pre-trained-models\bert-base-chinese", # pretrained model path on your disk
        "hidden_size": 768,
        "num_layers":1,
        "dropout":0.1,
    },
    
    "qwen_config":{
        "model_path":r"D:\pre-trained-models\Qwen-0.5B-Enhanced",
        "hidden_size": 1024,
        "dropout":0.1,
        "model_type": "qwen2",
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
    }
}


```

## 运行项目：
- run QA_system.py


---

### Gradio配置
```shell 
pip install -i https://mirrors.aliyun.com/pypi/simple/ gradio
```

- 注意，运行gradio程序时，do not open web proxy !!!
- if you do that, some func of gradio will not work

---
## 运行结果

![image](https://github.com/user-attachments/assets/33f55040-3de8-4cc7-b756-a7b3c30a3c99)
