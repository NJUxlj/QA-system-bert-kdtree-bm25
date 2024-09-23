import os
import json
import jieba
import numpy as np
from bm25 import BM25
from similarity_function import editing_distance, jaccard_distance
from kd_tree import KDTree
from gensim.models import Word2Vec # 获取词向量



from config import Config



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
        self.load_word2vec()
      elif algo == 'kdtree':
        self.load_kdtree()
      else:
        # 加载其他模型不需要做预处理
        pass 

  def load_bm25(self):
    pass
  
  
  def load_word2vec(self):
    if os.path.isfile("model.w2v"):
      self.w2v_model = Word2Vec.load("model.w2v")
    else:
      corpus = []# [sentence_num, word_num]
      for target, questions in self.target_to_questions.items():
        for question in questions:
          corpus.append(jieba.lcut(question))
          
      # 训练模型
      self.w2v_model = Word2Vec(corpus, vector_size=100, min_count=1)
      
      self.w2v_model.save("model.w2v")
      print("训练Word2Vec模型成功！")
      
    #借助词向量模型，将知识库中的问题向量化
    
    for target, questions in self.target_to_questions.items():
      pass
    
    
    
  def load_kdtree(self):
    pass
  
  
  def sentence_to_vector(self, sentence):
    vector = np.zeros(self.w2v_model.vector_size)
    
    count = 0
    for word in jieba.lcut(sentence):
      if word in self.w2v_model.wv:
        vector += self.w2v_model.wv[word]
        count+=1
    vector = vector/count
    
    # 做L2归一化，方便等下算cosine disitance
    vector = np.array(vector)/np.sqrt(np.sum(np.square(vector)))
    return vector
  
  def load_know_base(self, know_base_path):
    self.target_to_questions={} # 从标准问到问题集的映射字典
    with open(know_base_path, 'r', encoding = 'utf8') as f:
      for index, line in enumerate(f):
        line = json.loads(line)
        target = line['target']
        questions = line['questions']
        self.target_to_questions[target] = questions
    
    return 
  
  def query(self):
    pass
  
  
  
  
  
  
  
  
if __name__ == '__main__':
    qas = QASystem(Config['data_path'], "bm25")
    # question = "话费是否包月超了"
    # res = qas.query(question)
    # print(question)
    # print(res)
    
    while True:
        question = input("请输入问题：")
        res = qas.query(question)
        print("命中问题：", res)
        print("-----------")

  
