
from loader import load_data
import torch


class Evaluator(object):
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config['valid_data_path'], config, shuffle=False)
        self.stats_dict = {"correct":0, "wrong":0}

    
    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct":0, "wrong":0}  #清空前一轮的测试结果
        self.model.eval()
            
        for index, batch in enumerate(self.valid_data):
            if torch.cuda.is_available():
                [d.cuda for d in batch]
            
            input_id, labels = batch
            with torch.no_grad():
                predictions = self.model(input_id)
            self.write_stats(labels, predictions)
        
        self.show_stats()
        return
    
    
    
    
    def show_stats(self):
        pass
    
    
    
    
    def write_stats(self, labels, predictions):
        pass