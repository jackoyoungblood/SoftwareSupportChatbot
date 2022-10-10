import sys
from PyPDF2 import PdfReader
import os
import re
import openai

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
    question = 'Q: Can you list the steps to printing an individual label in imanage records manager?'
    toc_dict = get_toc(reader, 4, 8)
    page_nums = get_page_nums_with_keyword(toc_dict, keyword)
    content = get_content_from_pages(reader, page_nums)
    prompt = make_prompt(content, question)
    print(get_openai_response(prompt)) 