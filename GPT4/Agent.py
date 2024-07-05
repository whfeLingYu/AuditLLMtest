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

# 选项
choices = ["A", "B", "C", "D"]

# 确保你的环境变量已设置
os.environ["SERPAPI_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = "sk-"

# 定义提示模板
prompt_template = """
你是一个金融领域专家，你的任务是回答以下问题。请仔细阅读问题和选项，然后选择正确的答案，即A,B,C,D。
问题: {question}
选项: {choices}

回答:
"""

error_template = """
你是一个金融领域专家，你的任务是回答以下问题。请仔细阅读问题和选项，然后选择正确的答案，即A,B,C,D。
之前的回答过程中出现了错误，请重新尝试。
问题: {question}
选项: {choices}
错误信息: {error}

回答:
"""

# 初始化 OpenAI 的 ChatGPT 模型
llm = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-4o")
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)
# 加载工具
tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, handle_parsing_errors=True, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


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
    results = []
    dev_df = pd.read_csv(os.path.join("data", "test", "注册会计师（CPA)1.csv"), header=0, index_col=0)

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

        results.append({
            'question': question,
            'choices': choices_str,
            'actual_answer': actual_answer,
            'gpt_answer': gpt_answer,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct
        })

    result_df = pd.DataFrame(results)
    result_file = os.path.join("results", "agent_results_注册会计师（CPA）.csv")
    #如果文件不存在，创建文件
    if not os.path.exists("results"):
        os.makedirs("results")

    result_df.to_csv(result_file, index=False)
    print(f"Results saved to {result_file}")

    compute_accuracy(result_file)


if __name__ == "__main__":
    main()
