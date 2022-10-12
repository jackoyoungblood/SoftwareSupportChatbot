import sys
from PyPDF2 import PdfReader
import os
import re
import openai
import gradio as gr

#from transformers import AutoModelForCausalLM, AutoTokenizer
#import torch

#tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
#model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def get_toc(reader, toc_page_start, toc_page_end):
    toc_dict = {}
    regex = r'(?P<section>[\w+\s+]+) (?:\.+) (?P<page_num>\d+)'
    contents = reader.pages[toc_page_start - 1:toc_page_end]
    for page in contents:
        page_contents = page.extract_text().split('\n')
        for line in page_contents:
            m = re.search(regex, line)
            if m:
                groupdict = m.groupdict()
                toc_dict[groupdict['section'].lower()] = groupdict['page_num']
    return toc_dict

def get_page_nums_with_keyword(toc_dict, keyword):
    sections = list(filter(lambda section: keyword in section, toc_dict.keys()))
    return list(set([int(toc_dict[section]) for section in sections]))

def get_content_from_pages(reader, page_nums, page_limit = 3):
    content = ""
    for page_num in page_nums[:page_limit]:
        content += reader.pages[page_num - 1].extract_text()
    return content

def make_prompt(content, question, add_task_description = False):
    prompt=''
    prompt += content 
    if add_task_description:
        prompt += '\n'
        prompt += 'Answer the question as truthfully as possible and if you are unsure of the answer say "Sorry I don''t know"'
    prompt += '\n'
    prompt += question
    prompt += '\n'
    prompt += 'A:'
    return prompt

def get_openai_response(prompt):
    return openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.0,
        max_tokens=104,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

if __name__ == "__main__":
    reader = PdfReader("IRM_Web_Client_User_Guide_Legal_Version_10.3.3.pdf")
    keyword = "label"
    openai.api_key = os.getenv('OPENAIKEY')
    #question = 'Q: Can you list the steps to printing an individual label in imanage records manager?'
    toc_dict = get_toc(reader, 4, 8)
    page_nums = get_page_nums_with_keyword(toc_dict, keyword)
    content = get_content_from_pages(reader, page_nums)
    #prompt = make_prompt(content, question)
    #print(get_openai_response(prompt)) 

def predict(input, history=[]):
    # tokenize the new input sentence
    #new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    #bot_input_ids = torch.cat([torch.LongTensor(history), new_user_input_ids], dim=-1)

    # generate a response 
    #history = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id).tolist()

    # convert the tokens to text, and then split the responses into lines
    #response = tokenizer.decode(history[0]).split("<|endoftext|>")
    #response.remove("")
    response=''

    prompt = make_prompt(content, input)
    response=get_openai_response(prompt)
    responsetext = response['choices'][0]['text']
    # write some HTML
    #html = "<div class='chatbot'>"
    #for m, msg in enumerate(chattext):
    #    cls = "user" if m%2 == 0 else "bot"
    #    html += "<div class='msg {}'> {}</div>".format(cls, msg)
    #html += "<div class='msg {}'> {}</div>".format(cls, msg)
    #html += chattext
    #html += "</div>"
    
    #return html, history
    #responsetmp = history[0].split("Q:")
    #responsetmp = []
    #responsetmp = history.copy()
    #responsetmp.append(input)
    #responsetmp.append(responsetext)
    #history.append(input)
    #history.append(responsetext)
    history.append((input,responsetext))
    #response = [(responsetmp[i], responsetmp[i+1]) for i in range(0, len(responsetmp)-1, 2)]  # convert to tuples of list
    
    #history.append((response,responsetext))
    return history, history

css = """
.chatbox {display:flex;flex-direction:column}
.msg {padding:4px;margin-bottom:4px;border-radius:4px;width:80%}
.msg.user {background-color:cornflowerblue;color:white}
.msg.bot {background-color:lightgray;align-self:self-end}
.footer {display:none !important}
"""

#gr.Interface(fn=predict,
#             theme="default",
#             inputs=[gr.inputs.Textbox(placeholder="Submit your iManage Records Manager question."), "state"],
#             outputs=["html", "state"],
#             css=css,
#             examples=[["How do I view an electronic rendition in iManage Records Manager?"]]).launch()
gr.Interface(fn=predict,
             inputs=["text", "state"],
             outputs=["chatbot", "state"],
             examples=[["How do I print a file part label in imanage records manager?"]]).launch()
                     