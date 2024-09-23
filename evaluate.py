
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
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return
    
    
    
    
    def write_stats(self, labels, predictions):
        assert len(labels) == len(predictions)
        for label, prediction in zip(labels, predictions):
            prediction = torch.argmax(prediction,dim=1)
            
            if int(label) == int(prediction):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return
            
        