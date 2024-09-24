import gradio as gr
from transformers import pipeline
import numpy as np
import random

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




with gr.Blocks() as demo:
    section_labels = [
        "apple",
        "banana",
        "carrot",
        "donut",
        "eggplant",
        "fish",
        "grapes",
        "hamburger",
        "ice cream",
        "juice",
    ]

    with gr.Row():
        num_boxes = gr.Slider(0, 5, 2, step=1, label="Number of boxes")
        num_segments = gr.Slider(0, 5, 1, step=1, label="Number of segments")

    with gr.Row():
        img_input = gr.Image()
        img_output = gr.AnnotatedImage(
            color_map={"banana": "#a89a00", "carrot": "#ffae00"}
        )

    section_btn = gr.Button("Identify Sections")
    selected_section = gr.Textbox(label="Selected Section")

    def section(img, num_boxes, num_segments):
        sections = []
        for a in range(num_boxes):
            x = random.randint(0, img.shape[1])
            y = random.randint(0, img.shape[0])
            w = random.randint(0, img.shape[1] - x)
            h = random.randint(0, img.shape[0] - y)
            sections.append(((x, y, x + w, y + h), section_labels[a]))
        for b in range(num_segments):
            x = random.randint(0, img.shape[1])
            y = random.randint(0, img.shape[0])
            r = random.randint(0, min(x, y, img.shape[1] - x, img.shape[0] - y))
            mask = np.zeros(img.shape[:2])
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    dist_square = (i - y) ** 2 + (j - x) ** 2
                    if dist_square < r**2:
                        mask[i, j] = round((r**2 - dist_square) / r**2 * 4) / 4
            sections.append((mask, section_labels[b + num_boxes]))
        return (img, sections)

    section_btn.click(section, [img_input, num_boxes, num_segments], img_output)

    def select_section(evt: gr.SelectData):
        return section_labels[evt.index]

    img_output.select(select_section, None, selected_section)



demo.launch(share=True)


