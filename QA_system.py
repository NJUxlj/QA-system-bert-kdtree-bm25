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
    '''
     return BM25(self.corpus)
    '''
    self.corpus = {}
    for target, questions in self.target_to_questions.items():
      self.corpus[target] = []
      for question in questions:
        self.corpus[target]+= jieba.lcut(question) # 把questions中的所有词连成一个大列表
    
    self.bm25_model = BM25(self.corpus)
  
  
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
    
    self.target_to_vectors = {}
    
    for target, questions in self.target_to_questions.items():
      vectors = [] # [question_num, vector_size]
      for question in questions:
        vectors.append(self.sentence_to_vector(question))
      self.target_to_vectors[target] = np.array(vectors)
    
  def load_kdtree(self):
    pass
  
  
  def sentence_to_vector(self, sentence):
    '''
      return np.array(vector)
      对文本中所有词的embedding做pooling
    '''
    vector = np.zeros(self.w2v_model.vector_size)
    words = jieba.lcut(sentence)
    
    count = 0
    for word in words:
      if word in self.w2v_model.wv:
        vector += self.w2v_model.wv[word]
        count+=1
    vector = vector/count
    
    # 做L2归一化，方便等下算cosine disitance
    vector = np.array(vector)/np.sqrt(np.sum(np.square(vector)))
    return vector
  
  def load_know_base(self, know_base_path):
    self.target_to_questions={} # 从标准问到问题集的映射字典
    with open(know_base_path, encoding = 'utf8') as f:
      for index, line in enumerate(f):
        content = json.loads(line)
        target = content['target']
        questions = content['questions']
        self.target_to_questions[target] = questions
    
    return 
  
  def rank_using_bm25(self, results, top_k=5):
    '''
     用bm25做精排
    '''
  
  def recall_using_kv(self, results):
    '''
     用kv召回做粗排
    '''
  
  def query(self, user_query):
    results=[]
    
    if self.algo == 'editing_distance':
      for target, questions in self.target_to_questions.items():
        scores = [editing_distance(user_query, question) for question in questions]
        score = max(scores)
        results.append([target, score])
        
    elif self.algo=='jaccard_distance':
      for target, questions in self.target_to_questions.items():
        scores = [jaccard_distance(user_query, question) for question in questions]
        score = max(scores)
        results.append([target, score])
        
    elif self.algo=='bm25':
      words = jieba.lcut(user_query)
      results = self.bm25_model.get_scores(words)
    elif self.algo=='word2vec':
      query_vector = self.sentence_to_vector(user_query)
      for target, vectors in self.target_to_vectors.items():
        # user_query 和所有的相似问同时计算cosine similarity
        # [1, embedding_size] x [embedding_size, question_num]
        cos = query_vector.dot(vectors.transpose())
        print(cos)
        results.append([target, np.mean(cos)])
    elif self.algo == 'kdtree':
      pass
    else:
      assert "unknown algorithm!!!"
    
    
    # 这里应该过一轮召回，但是我们做了简化, 等召回写完了再加上
    sort_results = sorted(results, key=lambda x:x[1], reverse=True)
    
    # 这里做精排
    return sort_results[:3]
      
  
  
  
  
  
  
  
if __name__ == '__main__':
    qas = QASystem(Config["train_data_path"], "bm25")
    # question = "话费是否包月超了"
    # res = qas.query(question)
    # print(question)
    # print(res)
    
    while True:
        question = input("请输入问题：")
        res = qas.query(question)
        print("命中问题：", res)
        print("-----------")

  
