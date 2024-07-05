import re
import random
import os.path as osp
import numpy as np
import pandas as pd
import json
from collections import defaultdict

import os
# 选项
choices = ["A", "B", "C", "D"]

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
        # # 将 is_correct 列转换为布尔类型
        # df['is_correct'] = df['is_correct'].astype(str).str.lower() == 'true'
        # # 计算 is_correct 列中 True 的比例
        # correct_ratio = df['is_correct'].mean() * 100
        # result["is_correct_ratio"] = round(correct_ratio, 2)
        # print(f"is_correct ratio: {correct_ratio:.2f}%")

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
        print(f'result saved to {filename}')

if __name__ == '__main__':
    result_file = os.path.join("Llama2-13b-Chinese/results", "Llama3-Chinese-13B_results_注册会计师（CPA）.csv")
    compute_accuracy(result_file)
