# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      interfere
   Author :         zmfy
   DateTime :       2023/12/23 11:41
   Description :    
-------------------------------------------------
"""
from typing import Dict, List, Any
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import os
from tqdm import tqdm


class AlgSolution:

    def __init__(self):

        pre_seq_len = 384
        model_name = "/opt/data/private/LLM2/ChatGLM3/chatglm3-6b/"
        ptuning_path = f"./output/ptuing-chatglm3-6b-pt-20231220-{pre_seq_len}-2e-2/checkpoint-3500/"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.device = torch.device('cuda')

        if ptuning_path is not None:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, pre_seq_len=pre_seq_len)
            self.model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
            prefix_state_dict = torch.load(
                os.path.join(ptuning_path, "pytorch_model.bin"), map_location='cuda')
            new_prefix_state_dict = {}
            for k, v in prefix_state_dict.items():
                if k.startswith("transformer.prefix_encoder."):
                    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
            self.model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
            self.model = self.model.half().cuda()
            self.model.transformer.prefix_encoder.float()
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).cuda()
        self.model.eval()

    @staticmethod
    def pre_process(input_data: Dict) -> str:
        prompt = ("请你帮我判断如下小说简介来自哪一种类型的小说，"
                  "已知全部类型有'玄幻', '奇幻', '武侠', '仙侠', '都市', '历史', '游戏', '科幻', '奇闻异事', '古代言情', '现代言情', "
                  "'幻想言情'，小说简介如下《%s》，注意你的回复应该为'玄幻', '奇幻', '武侠', '仙侠', '都市', '历史', '游戏', '科幻', '奇闻异事', '古代言情', "
                  "'现代言情‘或'幻想言情'，不需要额外输出。") % (
                     input_data['input']).replace('\n', '')
        return prompt

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.device)
        # response = self.model.chat(self.tokenizer, prompt, history=[])
        max_len = 684
        response = self.model.generate(input_ids=inputs["input_ids"],
                                       max_length=max_len)
        response = response[0, inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        return response

    @staticmethod
    def post_process(response: str) -> str:
        return response

    def predicts(self, input_data: List[Dict], **kwargs) -> list[dict[str, str | Any]]:
        results = []
        for item in tqdm(input_data):
            if isinstance(item['id'], str) and item['id'].startswith('subject'):
                result = self.generate(item['input'])
            else:
                prompt = self.pre_process(item)
                response = self.generate(prompt)
                result = self.post_process(response)
            results.append({
                'id': item['id'],
                'input': item['input'],
                'output': result
            })
        return results

    def predict(self, text):
        prompt = self.pre_process({'input': text})
        response = self.generate(prompt)
        result = self.post_process(response)
        return result


if __name__ == '__main__':
    import json

    test = []
    with open('../data/full/test.jsonl') as f:
        for line in f:
            test.append(json.loads(line))
    print(f"测试集数量: {len(test)}")
    solution = AlgSolution()

    # # evaluation on test set
    # results = solution.predicts(test)
    # correct = 0
    # for i, res in enumerate(results):
    #     if res['output'] == test[i]['output']:
    #         correct += 1
    # print(f"准确率: {correct / len(test) :.4f}")

    # demo presentation
    demo = "如果天道无情，我便伐了这天!"
    result = solution.predict(demo)
    print(f"{demo} -> {result}")
    # 如果天道无情，我便伐了这天! -> 玄幻
