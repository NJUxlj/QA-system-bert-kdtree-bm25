# QA-system-bert-kdtree-bm25
一个基于FAQ知识库的问答系统，应用了多种文本匹配算法，包括：1.基于bert的表示型文本匹配模型。2.BM25匹配算法。3.KD树文本搜索算法


## Question-Answer System 的结构

- 用到的文本匹配算法
  - BM25
  - KD-Tree
- 用到的文本相似度预测模型
  - Bert
  - Bert+Text-RCNN+2xMLP

### FAQ知识库的结构




### BM25算法简介



### KD-Tree算法简介


### 表示型文本匹配模型



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
datasets
numpy
```

## 运行项目：
- 运行QA_system.py文件


---




## 运行结果