# 系统和模型的配置参数

Config = {
    "model_path": r"D:\pre-trained-models\bert-base-chinese",
    "schema_path": "../data/schema.json",
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
    }
}
