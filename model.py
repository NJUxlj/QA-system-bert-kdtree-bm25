from transformers import BertModel, BertTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn
from config import Config

'''
这里定义了两个模型

    BertMLP：直接将BERT的输出作为文本特征，使用MLP进行分类。
    BertTextRCNNMLP：将BERT的输出作为文本特征，使用RCNN进行特征提取，再使用MLP进行分类。


'''


class BertMLP(nn.Module):
    def __init__(self, bert_model_path, labels_count, hidden_dim=768, mlp_dim=256, dropout=0.1):
        super(BertMLP, self).__init__()  
        self.config = {  
            'bert_model_path': bert_model_path,  
            'labels_count': labels_count,  
            'hidden_dim': hidden_dim,  
            'mlp_dim': mlp_dim,  
            'dropout': dropout  
        }  
        
        
        
        self.bert = BertModel.from_pretrained(bert_model_path)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),  
            nn.ReLU(),  
            nn.Linear(mlp_dim, mlp_dim),  
            nn.ReLU(),  
            nn.Linear(mlp_dim, labels_count)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, input):
        sequence_output, _ = self.bert(input, return_dict=False) # [batch_size, seq_len, hidden_dim]
        

        output =  self.mlp(sequence_output[:, 0, :])
        
        output = self.softmax(output)
        
        return output





class BertTextRCNNMLP(nn.Module):
    
    '''
    模型结构：
     BERT + BiLSTM + TextCNN + MLP
    '''
    def __init__(self, bert_model_path, labels_count, rnn_hidden_dim=256, cnn_kernel_size=3,   
                 rnn_type='LSTM', num_layers=1, hidden_dim=768, mlp_dim=256, dropout=0.1):  
        super(BertTextRCNNMLP, self).__init__()  

        # 存储模型的超参数  
        self.config = {     
            'bert_model_path': bert_model_path,  
            'labels_count': labels_count,  
            'rnn_hidden_dim': rnn_hidden_dim,  
            'cnn_kernel_size': cnn_kernel_size,  
            'rnn_type': rnn_type,  
            'num_layers': num_layers,  
            'hidden_dim': hidden_dim,  
            'mlp_dim': mlp_dim,  
            'dropout': dropout  
        }  
        
        # bert默认将每一个input_id映射到768维向量
        self.bert = BertModel.from_pretrained(bert_model_path)  

        # 初始化RNN（可以是LSTM）        
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=rnn_hidden_dim,  
                           num_layers=num_layers, batch_first=True, bidirectional=True)  

        # 初始化CNN层用于特征提取  
        self.cnn = nn.Conv1d(in_channels=2 * rnn_hidden_dim,  # 因为是双向LSTM  
                             out_channels=rnn_hidden_dim,  
                             kernel_size=cnn_kernel_size,  
                             padding=cnn_kernel_size // 2)  

      
        self.dropout = nn.Dropout(dropout)  

        # MLP全连接层  
        self.mlp = nn.Sequential(  
            nn.Linear(rnn_hidden_dim, mlp_dim),  
            nn.ReLU(),  
            nn.Linear(mlp_dim, mlp_dim),  
            nn.ReLU(),  
            nn.Linear(mlp_dim, labels_count)  
        )  

        # Softmax层用于输出概率  
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, tokens):  
        '''  
        tokens: [batch_size, seq_len, hidden_dim]  
        masks: [batch_size, seq_len]  
        extras: [batch_size, extras_dim]  
        '''  
        # 通过BERT模型获取最后一个隐藏层的输出和池化输出  
        sequence_output, _ = self.bert(tokens, return_dict=False)  
        
        # 通过RNN  
        rnn_output, _ = self.rnn(sequence_output)  

        # 转换维度使其适应卷积层 (batch_size, channels, seq_len)  
        rnn_output = rnn_output.permute(0, 2, 1)  

        # 通过卷积层提取特征  
        cnn_output = self.cnn(rnn_output)  

        # 转换卷积输出维度  
        cnn_output = cnn_output.permute(0, 2, 1)   # (batch_size,  seq_len, channels)  

        # 提取最后一个时间步的特征作为文本特征  
        text_features = torch.mean(cnn_output, dim=1)    # (batch_size,  1, channels) 
        
        text_features = text_features.squeeze(1) # (batch_size,  channels) 

        dropout_output = self.dropout(text_features)  

        # concat_output = torch.cat([dropout_output, extras], dim=1)  

        mlp_output = self.mlp(dropout_output)  

        # 输出概率  
        proba = self.softmax(mlp_output)  

        return proba  


class QwenMLP(nn.Module):
    def __init__(self, qwen_model_path, labels_count, hidden_dim=1024, mlp_dim=256, dropout=0.1):
        super(QwenMLP, self).__init__()
        
        # qwen hidden_size = 1024
        self.qwen = AutoModelForSequenceClassification.from_pretrained(qwen_model_path) 
        
    
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, labels_count)
        )
        
        self.softmax = nn.Softmax(dim=1)
    
    
    def forward(self, input):
        sequence_output, _ = self.qwen(input, return_dict=False)
        output =  self.mlp(sequence_output[:, 0, :])
        output = self.softmax(output)
        return output
        
        
    


class BertQwenMLP(nn.Module):
    
    def __init__(self, bert_model_path, qwen_model_path, labels_count, hidden_dim=1024, mlp_dim=256, dropout=0.1):
        super(BertQwenMLP, self).__init__()
        # hidden_size = 768
        self.bert = BertModel.from_pretrained(bert_model_path)
        
        self.ffn = nn.Linear(768, 1024)
        
        # hidden_size = 1024
        self.qwen = AutoModelForSequenceClassification.from_pretrained(qwen_model_path) 

        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, labels_count)
        )
        
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input):
        sequence_output, _ = self.bert(input, return_dict=False)
        sequence_output = self.ffn(sequence_output)
        sequence_output, _ = self.qwen(sequence_output, return_dict=False)
        output =  self.mlp(sequence_output[:, 0, :])
        output = self.softmax(output)
        return output
    
    
    

def choose_optimizer(config, model):
    optimizer_type = config["optimizer"]
    
    if optimizer_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    if optimizer_type == 'sgd':
        return torch.optim.Adam(model.parameters(), lr =config['learning_rate'])

    raise Exception("Unknown optimizer type: ", optimizer_type)



if __name__ == '__main__':
    
    
    model1 = BertMLP(Config['bert_config']['model_path'], Config['labels_count'])
    input = torch.randint(0, 1000, (10, 50))
    logits = model1(input)
    print("logits = \n",logits)
    
    print("===================================")
    
    model = BertTextRCNNMLP(Config['bert_config']['model_path'], Config['labels_count'])
    
    input = torch.randint(0, 1000, (10, 50))
    logits = model(input)
    print("logits = \n",logits)
    
    pred = torch.argmax(logits, dim=1)
    print(pred)
    
    print("===================================")
    
    model2 = BertQwenMLP(Config['bert_config']['model_path'], Config['qwen_config']['model_path'], Config['labels_count'])
    
    input = torch.randint(0, 1000, (10, 50))
    logits = model(input)
    print("logits = \n",logits)
    
    pred = torch.argmax(logits, dim=1)
    print(pred)
    
