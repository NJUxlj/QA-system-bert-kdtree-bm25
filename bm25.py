import json
import math
import os
import pickle
import sys
from typing import Dict, List



'''
BM25算法公式

Score(Q,d)=\sum_i^n W_iR(q_i, d)

W_i = IDF(q_i) = log\frac{N-n(q_i)+0.5}{n(q_i)+0.5}

R(q_i,d) = \frac{f_i(k_1+1)}{f_i+K} \frac{qf_i(k_2+1)}{qf_i+k_2}

K = k_1(1-b+b\frac{dl}{avgdl})


q_i 为user_query中某词, f_i 为词频
k₁, k₂, b 为可调节常数, dl 为文档长度
avgdl 为所有文档平均长度

修改这些值，可以对 R(q_i, d)的值进行控制

'''

class BM25:
    EPSILON = 0.25
    PARAM_K1 = 1.5  # BM25算法中超参数
    PARAM_B = 0.6  # BM25算法中超参数
    PARAM_K2 = 1.2

    def __init__(self, corpus: Dict):
        """
            初始化BM25模型
            :param corpus: 文档集, 文档集合应该是字典形式，key为文档的唯一标识，val对应其文本内容，文本内容需要分词成列表
        """

        self.corpus_size = 0  # 文档数量
        self.wordNumsOfAllDoc = 0  
        self.doc_freqs = {}  # 记录每篇文档中查询词的词频  doc_id -> word -> count
        '''
            doc_freqs = {
                1: {
                    'word1': 1,
                    'word2': 1,
                    'word3': 1
                },
            }
        '''
        
        self.idf = {}  # 记录查询词的 IDF
        self.doc_len = {}  # 记录每篇文档的单词数  doc_id -> int
        self.docContainedWord = {}  # 包含单词 word 的文档集合
        self._initialize(corpus)


    def _initialize(self, corpus: Dict):
        """
            根据语料库构建倒排索引
            
            corpus: dict{
                1: ['word1', 'word2', 'word3'],
                2: ...
            }
        """
        # nd = {} # word -> number of documents containing the word
        for index, document in corpus.items():
            self.corpus_size += 1
            self.doc_len[index] = len(document)  # 文档的单词数
            self.wordNumsOfAllDoc += len(document)

            frequencies = {}  # 一篇文档中单词出现的频率
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs[index] = frequencies

            # 构建词到文档的倒排索引，将包含单词的和文档和包含关系进行反向映射
            '''
                frequencies = {
                    'word1': set(1, 2),
                    'word2': set(1, 2, 3),
                }
            '''
            for word in frequencies.keys():
                if word not in self.docContainedWord:
                    self.docContainedWord[word] = set()
                self.docContainedWord[word].add(index)

        # 计算 idf
        idf_sum = 0  # collect idf sum to calculate an average idf for epsilon value
        negative_idfs = []
        for word in self.docContainedWord.keys():
            doc_nums_contained_word = len(self.docContainedWord[word])
            idf = math.log(self.corpus_size - doc_nums_contained_word +
                           0.5) - math.log(doc_nums_contained_word + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)

        average_idf = float(idf_sum) / len(self.idf)
        eps = BM25.EPSILON * average_idf  
        for word in negative_idfs: # 如果idf小于0，则设置一个最小值eps，因为tf-idf一般是非负的
            self.idf[word] = eps

    @property
    def avgdl(self):
        '''
          return wordNumsOfAllDoc / corpus_size
        '''
        return float(self.wordNumsOfAllDoc) / self.corpus_size


    def get_score(self, query: List, doc_index):
        """
        计算查询 q 和文档 d 的相关性分数
        :param query: 查询词列表
        :param doc_index: 为语料库中某篇文档对应的索引
        """
        k1 = BM25.PARAM_K1
        b = BM25.PARAM_B
        k2 = BM25.PARAM_K2
        
        score = 0
        doc_freqs = self.doc_freqs[doc_index] # doc_id -> word -> count
        for word in query:
            if word not in doc_freqs: # 未登录词
                continue
            
            # idf(qi) * fi * (k1+1) / (fi + K)
            score += self.idf[word] * doc_freqs[word] * (k1 + 1) / (
                    doc_freqs[word] + k1 * (1 - b + b * self.doc_len[doc_index] / self.avgdl))
        return [doc_index, score]

    def get_scores(self, query):
        '''
         return [[doc_index, score], []...]
        
        '''
        scores = [self.get_score(query, index) for index in self.doc_len.keys()]
        return scores


if __name__ == "__main__":
    bm_model = BM25()
