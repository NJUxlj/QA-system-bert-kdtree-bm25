import gradio as gr
from transformers import pipeline
import numpy as np
import random
import time

from gradio import ChatMessage
from typing import List
from config import Config
from QA_system import QASystem

'''
Here is the UI of the QA system, which is a simple chatbot.
'''

#标题
title = "# FAQ问答系统-基于bert+qwen2的文本匹配"
#标题下的描述，支持md格式
description = "### 输入你想查询的问题后，点击submit按钮，系统会自动从FAQ库中挑出最匹配的答案，赶快试试吧！"
#输入样例
examples = [
    ["话费是否包月超了"],
    ["拨打哪个号码办理挂失"],
    ["查积分"]
    ]
#页面最后的信息，可以选择引用文章，支持md格式
article = "感兴趣的小伙伴可以阅读[github项目主页](https://github.com/NJUxlj/QA-system-bert-kdtree-bm25/tree/main)"


def get_faq_answer(user_query):
    '''
        定义执行函数
    '''
    
    qas = QASystem(Config["train_data_path"], "bm25")
    
    result: List[List[str, float]] = qas.query(user_query)
    answer = "用户问：\n" + user_query + "\n faq答案：\n"
    for index, ele in enumerate(result): 
            answer += f"Answer{index+1}" + ele[0] + "\n"
    score = np.mean(np.array([ele[1] for ele in result]))
    
    return answer, score


def clear_input():
    '''
        清除输入输出
    '''
    return " ", " ", " "

# 输入输出要与函数的输入输出个数一致
# demo = gr.Interface(
#     fn=get_faq_answer,inputs="text", outputs=[gr.Textbox(label="answer"), gr.Label(label="score")], 
#              title=title, description=description, examples=examples, article=article
# )


#构建Blocks上下文
with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Column():    # 列排列
        # context = gr.Textbox(label="context")
        user_query = gr.Textbox(label="user_query")
    with gr.Row():       # 行排列
        clear = gr.Button("clear")
        submit = gr.Button("submit")
    with gr.Column():    # 列排列
        answer = gr.Textbox(label="answer")
        score = gr.Label(label="score")
    #绑定submit点击函数
    submit.click(fn=get_faq_answer, inputs=user_query, outputs=[answer, score])
    # 绑定clear点击函数
    clear.click(fn=clear_input, inputs=[], outputs=[user_query, answer, score])
    gr.Examples(examples, inputs=user_query)
    gr.Markdown(article)



demo.launch(share=True)