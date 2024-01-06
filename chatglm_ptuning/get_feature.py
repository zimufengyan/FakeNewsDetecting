# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      get_feature
   Author :         zmfy
   DateTime :       2023/12/19 16:35
   Description :    
-------------------------------------------------
"""
# 读取特征
# 制作新的数据
# 保存数据
import os
import json
import pandas as pd
import random
from tqdm import tqdm

from baidu_text_trans import BaiduTextTranslator


def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    # print(df['label'].value_counts())   # 3171 REAL and 3164 Fake ones
    df.columns = ['unnamed', 'title', 'text', 'label']
    df['input'] = 'Title: ' + df['title'] + '. Body: ' + df['text'].replace('\n', ' ')
    df['output'] = df['label'].str.lower()
    df = df.drop(['title', 'text', 'label', 'unnamed'], axis=1)
    records = df.to_dict('records')

    n_samples = len(records)
    test_size = int(n_samples * 0.2)

    # 随机选择测试集样本索引
    test_idxs = random.sample(range(n_samples), test_size)
    train_idxs = list(set(range(n_samples)) - set(test_idxs))

    # 分割数据
    train = [records[i] for i in train_idxs]
    test = [records[i] for i in test_idxs]
    return train, test


def translate_back_augment(text, langs=None):
    """
    回译法进行数据增强
    :param text: 原文
    :param langs: 回译语言
    :return: 回译文
    """
    if langs is None:
        langs = ['en', 'fra', 'ru', 'en']
    translator = BaiduTextTranslator()
    for i in range(len(langs) - 1):
        src = langs[i]
        dst = langs[i+1]
        text = translator.trans(text, from_lang=src, to_lang=dst)
        if text == "":
            return ""
    return text


def crop_sentence(sentence, max_len=1024):
    # 将句子分割成单词
    words = sentence.split()

    # 如果单词数不超过max_len，则返回原句子
    if len(words) <= max_len:
        return sentence

    # 找到离第max_len个单词最近的句号的位置
    index = 0
    for i in range(max_len):
        if words[max_len - i - 1].endswith('.'):
            index = max_len - i - 1
            break

    # 截取句子
    cropped_sentence = ' '.join(words[:index + 1])
    return cropped_sentence


def add_feature(input_text):
    prompt = ('Now you are a media data researcher and you are working on fake news detection, '
              'determine whether the following news is real or fake: %s\nNote that you only '
              'need to return "real" or "fake", no additional output is required.') % (
                 input_text).replace('\n', '')
    return prompt


if __name__ == "__main__":
    # 用法示例：
    news_file_path = '../data/news.csv'

    # 构建特征数据
    os.makedirs("../data", exist_ok=True)
    train_json_file = "../data/train.json"
    test_json_file = "../data/test.json"

    train_jsonl_content, test_jsonl_content = read_csv_file(news_file_path)
    max_prompt_len = 0

    for idx, content in enumerate(train_jsonl_content):
        content['id'] = idx
        content['input'] = add_feature(crop_sentence(content['input']))
        tokens = content['input'].strip().split()
        if len(tokens) > max_prompt_len:
            max_prompt_len = len(tokens)

    # augment
    # 随机选择待增强数据
    n_samples = len(train_jsonl_content)
    aug_size = int(n_samples * 0.3)
    aug_idx = random.sample(range(n_samples), aug_size)
    temp = [train_jsonl_content[i] for i in aug_idx]
    augmented_contents = []
    # max_idx = train_jsonl_content[-1]['id']
    # loop = tqdm(enumerate(temp), total=len(temp))
    # for idx, content in loop:
    #     aug_text = translate_back_augment(content['input'])
    #     if aug_text == "":
    #         continue
    #     aug_text = add_feature(aug_text)
    #     aug_content = {'id': max_idx+idx+1, 'input': aug_text, 'output': content['output']}
    #     augmented_contents.append(aug_content)
    #     if len(aug_text) > max_prompt_len:
    #         max_prompt_len = len(aug_text)
    #     loop.set_description(f'Augmentation')
    # train_jsonl_content += augmented_contents

    with open(train_json_file, 'w', encoding='utf-8') as file:
        json.dump(train_jsonl_content, file, ensure_ascii=False, indent=4)

    print("train data done!")

    for idx, content in enumerate(test_jsonl_content):
        content['id'] = idx
        content['input'] = add_feature(crop_sentence(content['input']))
        tokens = content['input'].strip().split()
        if len(tokens) > max_prompt_len:
            max_prompt_len = len(tokens)

    with open(test_json_file, 'w', encoding='utf-8') as file:
        json.dump(test_jsonl_content, file, ensure_ascii=False, indent=4)

    print("val data done!")

    print(f"train size: {len(train_jsonl_content)}, including augmented data with size: {len(augmented_contents)}; "
          f"test size: {len(test_jsonl_content)}; max prompt length: {max_prompt_len}")

