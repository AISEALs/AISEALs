from transformers import AutoModel, AutoTokenizer, AutoConfig
import gradio as gr
import time
import torch
import os

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, cache_dir="/group/30147/summerwuxia/model/chatglm-6b")
# 使用chatglm ptuning的结果进行推理
config = AutoConfig.from_pretrained("/group/30147/summerwuxia/model/chatglm-6b/models--THUDM--chatglm-6b/snapshots/aa51e62ddc9c9f334858b0af44cf59b05c70148a", trust_remote_code=True)
#config.pre_seq_len = 64
config.pre_seq_len = 128
model = AutoModel.from_pretrained("/group/30147/summerwuxia/model/chatglm-6b/models--THUDM--chatglm-6b/snapshots/aa51e62ddc9c9f334858b0af44cf59b05c70148a", config=config, trust_remote_code=True)
#prefix_state_dict = torch.load(os.path.join("/group/30147/summerwuxia/model/checkpoint/adgen-chatglm-6b/adgen-chatglm-6b-ft-1e-4/checkpoint-28000", "pytorch_model.bin"))
prefix_state_dict = torch.load(os.path.join("/group/30147/summerwuxia/model/checkpoint/adgen-chatglm-6b/adgen-chatglm-6b-pt-128-2e-2/checkpoint-3000", "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

model = model.half().cuda()
model = model.eval()

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2


def predict(input, max_length, top_p, temperature, history=None):
    if history is None:
        history = []
    start_time = time.time()
    for response, history, tokens_sum in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        updates = []
        for query, response in history:
            updates.append(gr.update(visible=True, value="用户：" + query))
            updates.append(gr.update(visible=True, value="ChatGLM-6B：" + response))
            updates.append(gr.update(visible=True, value="已使用tokens数：" + str(tokens_sum)))
        if len(updates) < MAX_BOXES:
            updates = updates + [gr.Textbox.update(visible=False)] * (MAX_BOXES - len(updates))
        
        yield [history] + updates
    


with gr.Blocks() as demo:
    state = gr.State([])
    text_boxes = []
    for i in range(MAX_BOXES):
        if i % 2 == 0:
            text_boxes.append(gr.Markdown(visible=False, label="提问："))
        else:
            text_boxes.append(gr.Markdown(visible=False, label="回复："))

    with gr.Row():
        with gr.Column(scale=4):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", lines=11).style(
                container=False)
            with gr.Accordion("推理速度显示："):
                gr.Markdown("Look at me...")
        with gr.Column(scale=1):
            max_length = gr.Slider(0, 8192, value=4096, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
            button = gr.Button("Generate")
    button.click(predict, [txt, max_length, top_p, temperature, state], [state] + text_boxes)
    
demo.queue().launch(server_name='0.0.0.0', share=False, server_port=7999)
