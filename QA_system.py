import os
import json
import jieba
import numpy as np
from bm25 import BM25
from similarity_function import editing_distance, jaccard_distance
from kd_tree import KDTree
from gensim.models import Word2Vec # 获取词向量




class QASystem():
  def __init__(self, know_base_path, algo):
      '''
      :param know_base_path: 知识库文件路径
      :param algo: 选择不同的算法
      '''
      self.load_know_base(know_base_path)
      self.algo = algo
      if algo == 'bm25':
        self.load_bm25()
      elif algo == 'word2vec':
        self.load_word2vec
      else:
        pass 

  def load_bm25():
    pass

  
