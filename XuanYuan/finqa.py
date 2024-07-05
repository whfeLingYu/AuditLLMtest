import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
model_name_or_path = "/home/zhupengyu/Duxiaoman-DI/XuanYuan-13B-Chat"
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, use_fast=False, legacy=True)
model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16,device_map="auto")
model.eval()
print("模型加载完成")



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
# 定义提示模板
prompt_template = """
你是一个金融领域专家，你的任务是回答以下问题。请仔细阅读问题和选项，然后选择正确的答案，只需要给出选项即可，即A,B,C,D，不要给出任何解析过程。
问题: {question}
选项: {choices}

回答:
"""

error_template = """
你是一个金融领域专家，你的任务是回答以下问题。请仔细阅读问题和选项，然后选择正确的答案，只需要给出选项即可，即A,B,C,D，不要给出任何解析过程。
之前的回答过程中出现了错误，请重新尝试。
问题: {question}
选项: {choices}
错误信息: {error}

回答:
"""

def get_answer_from_gpt(question, choices):
    # agent = initialize_new_agent()
    prompt = prompt_template.format(question=question, choices=choices)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    answer = model.generate(**inputs, max_new_tokens=600, repetition_penalty=1.1)
    answer = tokenizer.decode(answer.cpu()[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print(answer)
    print("===")
    return answer

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
        df = pd.read_csv(result_file, names=['question', 'choices', 'actual_answer', 'gpt_answer', 'predicted_answer', 'is_correct'], index_col=False)
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
    filename = osp.join(os.path.dirname(result_file), 'result.json')
    with open(filename, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
        print(f'result save to {filename}')

def main():
    results = []
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

        results.append({
            'question': question,
            'choices': choices_str,
            'actual_answer': actual_answer,
            'gpt_answer': gpt_answer,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct
        })

    result_df = pd.DataFrame(results)
    result_file = os.path.join("results", "results_注册会计师（CPA）.csv")
    #如果文件不存在，创建文件
    if not os.path.exists("results"):
        os.makedirs("results")

    result_df.to_csv(result_file, index=False)
    print(f"Results saved to {result_file}")

    compute_accuracy(result_file)

if __name__ == "__main__":
    main()
