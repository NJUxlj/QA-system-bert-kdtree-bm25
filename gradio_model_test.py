import gradio as gr
from transformers import pipeline
import numpy as np
import random
import time
from config import Config

from gradio import ChatMessage

# # 文本分类, bert默认是18分类
# demo = gr.Interface.from_pipeline(pipeline(task="text-classification", 
#                                            model = Config['bert_config']['model_path']))


# demo.launch(share=True)


#标题
title = "抽取式问答"
#标题下的描述，支持md格式
description = "### 输入上下文与问题后，点击submit按钮，可从上下文中抽取出答案，赶快试试吧！"
#输入样例
examples = [
    ["普希金从那里学习人民的语言，吸取了许多有益的养料，这一切对普希金后来的创作产生了很大的影响。这两年里，普希金创作了不少优秀的作品，如《囚徒》、《致大海》、《致凯恩》和《假如生活欺骗了你》等几十首抒情诗，叙事诗《努林伯爵》，历史剧《鲍里斯·戈都诺夫》，以及《叶甫盖尼·奥涅金》前六章。", "著名诗歌《假如生活欺骗了你》的作者是"],
    ["普希金从那里学习人民的语言，吸取了许多有益的养料，这一切对普希金后来的创作产生了很大的影响。这两年里，普希金创作了不少优秀的作品，如《囚徒》、《致大海》、《致凯恩》和《假如生活欺骗了你》等几十首抒情诗，叙事诗《努林伯爵》，历史剧《鲍里斯·戈都诺夫》，以及《叶甫盖尼·奥涅金》前六章。", "普希金创作的叙事诗叫什么"]
    ]
#页面最后的信息，可以选择引用文章，支持md格式
article = "感兴趣的小伙伴可以阅读[gradio专栏](https://blog.csdn.net/sinat_39620217/category_12298724.html?spm=1001.2014.3001.5482)"

gr.Interface.from_pipeline(
    pipeline("question-answering", model=Config['bert_config']['model_path']),
    title=title, description=description, examples=examples, article=article).launch()
                        

