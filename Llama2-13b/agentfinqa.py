import os
import re
import glob
import random
import os.path as osp
import numpy as np
import openai
import pandas as pd
import json
from collections import defaultdict
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import load_tools, initialize_agent, AgentType
from openai import OpenAI
import csv

# 选项
choices = ["A", "B", "C", "D"]

# 确保你的环境变量已设置
serpapi_keys = [
    "",
    "",
    "",
    ""
    ""
]
current_key_index = 0
os.environ["SERPAPI_API_KEY"] = serpapi_keys[current_key_index]
api_key = "EMPTY"
# 修改为你启动api-for-open-llm项目所在的服务地址和端口
api_url = "http://0.0.0.0:8000/v1"
modal= "Llama-2-13b-chat-hf"
# 定义提示模板
prompt_template = """
You are an expert in the field of Chinese finance and your task is to answer the following questions. Please read the questions and choices carefully and then choose the correct answers, i.e. A,B,C,D.
clquestion: {question}
choices: {choices}

Note that only Search or Calculator can be selected in the Action,Final Answer and Action cannot coexist
Action:Search
Action Input: Specifies the content
perhaps
Action:Calculator
Action Input: Specifies the content

Finally if you get the answer, you only need to give the options, such as A,B,C,D,Don't add any other words.clearthe format must be as follows:
Final Answer: Options

Because of the limited number of words, you must give the answer as soon as possible
Answer:
"""

error_template = """
You are an expert in the field of Chinese finance and your task is to answer the following questions. Please read the questions and choices carefully and then choose the correct answers, i.e. A,B,C,D.
There was an error in the previous answer, please try again.
question: {question}
choices: {choices}
error message: {error}

Note that only Search or Calculator can be selected in the Action,Final Answer and Action cannot coexist
Action:Search
Action Input: Specifies the content
perhaps
Action:Calculator
Action Input: Specifies the content

Finally if you get the answer, you only need to give the options, such as A,B,C,D,Don't add any other words.clearthe format must be as follows:
Final Answer: Options

Because of the limited number of words, you must give the answer as soon as possible
Answer:
"""
def _handle_error(error) -> str:
    return str(error)[:50]

from langchain.memory import ConversationBufferMemory
# 初始化 OpenAI 的 ChatGPT 模型
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(model_name=modal, openai_api_key=api_key, openai_api_base=api_url)
# 加载工具
tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, handle_parsing_errors=_handle_error, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,memory=memory, verbose=True)

def switch_serpapi_key():
    global current_key_index
    current_key_index = (current_key_index + 1) % len(serpapi_keys)
    os.environ["SERPAPI_API_KEY"] = serpapi_keys[current_key_index]
    print(f"Switched to SERPAPI_API_KEY: {serpapi_keys[current_key_index]}")

def get_answer_from_gpt(question, choices):
    global agent
    prompt = prompt_template.format(question=question, choices=choices)
    error_prompt_message = None
    while True:
        try:
            if error_prompt_message:
                response = agent.run(error_prompt_message)
                return response.strip()
            else:
                response = agent.run(prompt)
                return response.strip()
        except Exception as e:
            error_message = f"Error occurred: {e}"
            print(error_message)
            if "Got error from SerpAPI" in str(e):
                switch_serpapi_key()
                agent = initialize_agent(tools, llm, handle_parsing_errors=True,agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory,verbose=True)
            else:
                try:
                    error_prompt_message = error_template.format(question=question, choices=choices, error=error_message)
                    response = agent.run(error_prompt_message)
                    return response.strip()
                except Exception as retry_e:
                    print(f"Error during retry: {retry_e}")
                    error_message += f" | Retry error: {retry_e}"
                    error_prompt_message = error_template.format(question=question, choices=choices, error=error_message)

def check_answer(predicted, actual):
    return predicted == actual

def extract_choice(response):
    '''
        Always return a choice, even cannot match by regex,
        to ensure fair comparison to other models.
    '''
    response = str(response)
    if response[0] in choices:
        return response[0]
    patterns = [
        (r'答案(选项)?(是|为)：? ?([ABCD])', 3),
        (r'答案(是|为)选项 ?([ABCD])', 2),
        (r'故?选择?：? ?([ABCD])', 1),
        (r'([ABCD]) ?选?项(是|为)?正确', 1),
        (r'正确的?选项(是|为) ?([ABCD])', 2),
        (r'答案(应该)?(是|为)([ABCD])', 3),
        (r'选项 ?([ABCD]) ?(是|为)?正确', 1),
        (r'选择答案 ?([ABCD])', 1),
        (r'答案?：?([ABCD])', 1),
        (r'([ABCD])(选?项)?是?符合题意', 1),
        (r'答案选项：? ?([ABCD])', 1),
        (r'答案(选项)?为(.*?)([ABCD])', 3),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return answer
    patterns = [
        (r'([ABCD])(.*?)当选', 1),
        (r'([ABCD])(.*?)正确', 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            while m:
                answer = m.group(idx)
                m = re.search(pattern, m.group(0)[1:], re.M)
            assert answer in choices
            return answer
    patterns = [
        (r'[^不]是：? ?([ABCD])', 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return answer
    pattern = r'^[^ABCD]*([ABCD])[^ABCD]*$'
    m = re.match(pattern, response)
    if m:
        answer = m.group(1)
        assert answer in choices
        return answer
    return choices[random.randint(0, 3)]

def compute_accuracy(result_file):
    all_acc = defaultdict(float)
    result = {}

    try:
        df = pd.read_csv(result_file,
                         names=['question', 'choices', 'actual_answer', 'gpt_answer', 'predicted_answer', 'is_correct'],
                         index_col=False)
        if df.iloc[0]['question'] == '1':
            df = df.drop(0)
        df['pred'] = df['gpt_answer'].apply(extract_choice)
        df['acc'] = df['actual_answer'] == df['pred']
        acc = np.mean(df['acc']) * 100
        all_acc["注册会计师（CPA）"] = acc
        result["注册会计师（CPA）"] = round(acc, 2)
    except Exception as e:
        print(f"Error computing accuracy: {e}")
        return

    print(f"注册会计师（CPA）: {all_acc['注册会计师（CPA）']:.2f}")
    avg_all_acc = np.mean(list(all_acc.values()))
    print(f"{'Overall':30s} {avg_all_acc:.2f}")
    result['Overall'] = round(avg_all_acc, 2)
    filename = osp.join(os.path.dirname(result_file), 'agent_result.json')
    with open(filename, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
        print(f'result save to {filename}')

def main():
    result_file = os.path.join("results", "agent_results_注册会计师（CPA）.csv")
    # 如果文件不存在，创建文件
    if not os.path.exists("results"):
        os.makedirs("results")

    with open(result_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['question', 'choices', 'actual_answer', 'gpt_answer', 'predicted_answer', 'is_correct'])

        dev_df = pd.read_csv(os.path.join("data", "test", "注册会计师（CPA）.csv"), header=0, index_col=0)

        for idx, row in dev_df.iterrows():
            question = row['Question']
            choices_str = "\n".join([f"{choice}. {row[choice]}" for choice in choices])
            actual_answer = row['Answer']
            gpt_answer = get_answer_from_gpt(question, choices_str)

            if gpt_answer:
                predicted_answer = extract_choice(gpt_answer)
                is_correct = check_answer(predicted_answer, actual_answer)
            else:
                predicted_answer = None
                is_correct = False

            writer.writerow([question, choices_str, actual_answer, gpt_answer, predicted_answer, is_correct])
            file.flush()  # 确保每次写入后都将缓冲区内容写入文件

    print(f"Results saved to {result_file}")

    compute_accuracy(result_file)

if __name__ == "__main__":
    main()