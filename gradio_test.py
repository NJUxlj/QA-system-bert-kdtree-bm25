import gradio as gr
from transformers import pipeline
import numpy as np
import random
import time

from gradio import ChatMessage

'''
Here is the UI of the QA system, which is a simple chatbot.
'''


def reverse(text):
    return text[::-1]


def echo(message, history):
    '''
     回显用户消息
    '''
    return message




# demo = gr.Interface(reverse, "text", "text")
# demo.launch(share=True, auth=("username", "password"))


# pipe = pipeline("image-classification")
# demo = gr.Interface.from_pipeline(pipe)
# demo.launch()


# demo = gr.ChatInterface(fn = echo, examples = ['BM25','KD-tree'], title = 'echo bot')
# demo.launch()



# # 消息回显函数
# def yes(message, history):
#     return "yes"

# def vote(data: gr.LikeData):
#     if data.liked:
#         print("You upvoted this response: " + data.value["value"])
#     else:
#         print("You downvoted this response: " + data.value["value"])

# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot(placeholder="<strong>Your Personal Yes-Man</strong><br>Ask Me Anything")
#     chatbot.like(vote, None, None)
#     gr.ChatInterface(
#         fn=yes, 
#         chatbot=chatbot,
#         examples = [{"text":"BM25"}, {"text":"KD-tree"}, {"text":"KNN"}],
#         title = "Echo Bot",
#         multimodal=True
#         )
    
# demo.launch(share=True, auth=("username", "password"))










# user_query = gr.Interface(lambda name: name, "text", "text")

# matched_faq_questions = gr.Interface(lambda name: name, "text", "text")


# demo=gr.TabbedInterface([user_query, matched_faq_questions], ["User Query", "FAQ questions"])
# demo.queue(max_size=10)

# demo.launch(share = True)








# def update(name):
#     return f"Welcome to Gradio, {name}!"

# with gr.Blocks() as demo:
#     gr.Markdown(
#         """
#             # Hello World!
#             Start typing below to see the output.
#         """
#         )
#     with gr.Row():
#         inp = gr.Textbox(placeholder="What is your name?")
#         out = gr.Textbox()
#     btn = gr.Button("Run")
#     btn.click(fn=update, inputs=inp, outputs=out)

# demo.launch(share=True)




# def welcome(name):
#     return f"Welcome to Gradio, {name}!"

# with gr.Blocks() as demo:
#     gr.Markdown(
#     """
#     # Hello World!
#         Start typing below to see the output.
#     """)
#     inp = gr.Textbox(placeholder="What is your name?")
#     out = gr.Textbox()
#     inp.change(welcome, inp, out)


# demo.launch(share=True)







# def flip_text(x):
#     return x[::-1]

# def flip_image(x):
#     return np.fliplr(x)

# with gr.Blocks() as demo:
#     gr.Markdown("Flip text or image files using this demo.")
#     with gr.Tab("Flip Text"):
#         text_input = gr.Textbox()
#         text_output = gr.Textbox()
#         text_button = gr.Button("Flip")
#     with gr.Tab("Flip Image"):
#         with gr.Row():
#             image_input = gr.Image()
#             image_output = gr.Image()
#         image_button = gr.Button("Flip")

#     with gr.Accordion("Open for More!", open=False):
#         gr.Markdown("Look at me...")
#         temp_slider = gr.Slider(
#             0, 1,
#             value=0.1,
#             step=0.1,
#             interactive=True,
#             label="Slide me",
#         )

#     text_button.click(flip_text, inputs=text_input, outputs=text_output)
#     image_button.click(flip_image, inputs=image_input, outputs=image_output)


# demo.launch(share = True, auth=("username", "password"))





# with gr.Blocks() as demo:
#     input_text = gr.Textbox()

#     @gr.render(inputs=input_text)
#     def show_split(text):
#         if len(text) == 0:
#             gr.Markdown("## No Input Provided")
#         else:
#             for letter in text:
#                 with gr.Row():
#                     text = gr.Textbox(letter)
#                     btn = gr.Button("Clear")
#                     btn.click(lambda: gr.Textbox(value=""), None, text)





# with gr.Blocks() as demo:
#     with gr.Row():
#         with gr.Column(scale=1):
#             text1 = gr.Textbox()
#             text2 = gr.Textbox()
#         with gr.Column(scale=4):
#             btn1 = gr.Button("Button 1")
#             btn2 = gr.Button("Button 2")
#         with gr.Column(scale = 6):
#             with gr.Group():
#                 gr.Textbox(label="First")
#                 gr.Textbox(label="Last")



# with gr.Blocks() as demo:
#     with gr.Tab("Lion"):
#         gr.Image("lion.jpg")
#         gr.Button("New Lion")
#     with gr.Tab("Tiger"):
#         gr.Image("tiger.jpg")
#         gr.Button("New Tiger")




# with gr.Blocks() as demo:
#     section_labels = [
#         "apple",
#         "banana",
#         "carrot",
#         "donut",
#         "eggplant",
#         "fish",
#         "grapes",
#         "hamburger",
#         "ice cream",
#         "juice",
#     ]

#     with gr.Row():
#         num_boxes = gr.Slider(0, 5, 2, step=1, label="Number of boxes")
#         num_segments = gr.Slider(0, 5, 1, step=1, label="Number of segments")

#     with gr.Row():
#         img_input = gr.Image()
#         img_output = gr.AnnotatedImage(
#             color_map={"banana": "#a89a00", "carrot": "#ffae00"}
#         )

#     section_btn = gr.Button("Identify Sections")
#     selected_section = gr.Textbox(label="Selected Section")

#     def section(img, num_boxes, num_segments):
#         sections = []
#         for a in range(num_boxes):
#             x = random.randint(0, img.shape[1])
#             y = random.randint(0, img.shape[0])
#             w = random.randint(0, img.shape[1] - x)
#             h = random.randint(0, img.shape[0] - y)
#             sections.append(((x, y, x + w, y + h), section_labels[a]))
#         for b in range(num_segments):
#             x = random.randint(0, img.shape[1])
#             y = random.randint(0, img.shape[0])
#             r = random.randint(0, min(x, y, img.shape[1] - x, img.shape[0] - y))
#             mask = np.zeros(img.shape[:2])
#             for i in range(img.shape[0]):
#                 for j in range(img.shape[1]):
#                     dist_square = (i - y) ** 2 + (j - x) ** 2
#                     if dist_square < r**2:
#                         mask[i, j] = round((r**2 - dist_square) / r**2 * 4) / 4
#             sections.append((mask, section_labels[b + num_boxes]))
#         return (img, sections)

#     section_btn.click(section, [img_input, num_boxes, num_segments], img_output)

#     def select_section(evt: gr.SelectData):
#         return section_labels[evt.index]

#     img_output.select(select_section, None, selected_section)








# def generate_response(history):
#     history.append(
#         ChatMessage(role="assistant",
#                     content="How can I help you?")
#         )
#     return history


# def generate_response(history):
#     history.append(
#         ChatMessage(role="assistant",
#                     content="The weather API says it is 20 degrees Celcius in New York.",
#                     metadata={"title": "🛠️ Used tool Weather API"})
#         )
#     return history





# def load():
#     return [
#         ("Here's an audio", gr.Audio("https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav")),
#         ("Here's an video", gr.Video("https://github.com/gradio-app/gradio/raw/main/demo/video_component/files/world.mp4"))
#     ]

# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot()
#     button = gr.Button("Load audio and video")
#     button.click(load, None, chatbot)




# demo.launch(share=True)






# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot()
#     msg = gr.Textbox()
#     clear = gr.ClearButton([msg, chatbot])

#     def respond(message, chat_history):
#         bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
#         chat_history.append((message, bot_message))
#         time.sleep(2)
#         return "", chat_history

#     msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    
    
# demo.launch(share=True)








# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.

# 用于后续对不同类别的文本进行颜色标记。
color_map = {
    "harmful": "crimson",
    "neutral": "gray",
    "beneficial": "green",
}

def html_src(harm_level):
    return f"""
<div style="display: flex; gap: 5px;">
  <div style="background-color: {color_map[harm_level]}; padding: 2px; border-radius: 5px;">
  {harm_level}
  </div>
</div>
"""

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    '''
       定义一个函数来向 history 添加消息。history 是对话历史的列表。
       
        迭代 message["files"]: 如果有文件，则将文件添加到 history 中。
        添加文本消息: 如果消息有文本内容，也会添加到 history 中。
        
        return history, gr.MultimodalTextbox
    '''
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history, response_type):
    '''
      处理聊天机器人的不同响应类型
    '''
    if response_type == "gallery":
        history[-1][1] = gr.Gallery(
            [
                "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
                "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
            ]
        )
    elif response_type == "image":
        history[-1][1] = gr.Image(
            "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"
        )
    elif response_type == "video":
        history[-1][1] = gr.Video(
            "https://github.com/gradio-app/gradio/raw/main/demo/video_component/files/world.mp4"
        )
    elif response_type == "audio":
        history[-1][1] = gr.Audio(
            "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav"
        )
    elif response_type == "html":
        history[-1][1] = gr.HTML(
            html_src(random.choice(["harmful", "neutral", "beneficial"]))
        )
    else:
        history[-1][1] = "Cool!"
    return history

with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        scale=1,
    )
    response_type = gr.Radio(
        [
            "image",
            "text",
            "gallery",
            "video",
            "audio",
            "html",
        ],
        value="text",
        label="Response Type",
    )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        placeholder="Enter message or upload file...",
        show_label=False,
    )
    
    # 用户往 chat_input 中输入的内容会通过 add_message 函数处理，然后添加返回的history到 chatbot 中，最后将chat_input置为空。
    # [chat_bot, chat_input] 会以某种形式传入add_message函数, add_message函数返回 [history, empty TextBox(interactive = False)]
    # 将消息更新至聊天板，并重置输入框。
    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input]
    )
    
    # 输入 [chatbot, response_type]: 这两个组件的当前状态作为输入传递给 bot 函数。
    # 输出更新在 chatbot: 将生成的响应更新到 chatbot 中，使得用户能够看到机器人的回复。
    bot_msg = chat_msg.then(
        bot, [chatbot, response_type], chatbot, api_name="bot_response"
    )
    
    
    # 更新 chat_input: 最终它重置了 chat_input，准备接收下一个输入，这样用户在上一个响应展示完之后可以立即开始输入新的消息。
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    chatbot.like(print_like_dislike, None, None)
    
demo.launch(share=True)