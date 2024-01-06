# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      baidu_text_trans
   Author :         zmfy
   DateTime :       2024/1/5 15:41
   Description :    
-------------------------------------------------
"""
import requests
import random
import json
from hashlib import md5


class BaiduTextTranslator(object):
    def __init__(self, appid=None, appkey=None):
        self.appid = appid if appid is not None else "20240105001931995"
        self.appkey = appkey if appkey is not None else "z1CzvK9Mkwztmt0gkoiA"

        endpoint = 'http://api.fanyi.baidu.com'
        path = '/api/trans/vip/translate'
        self.url = endpoint + path

    def trans(self, query, from_lang='en', to_lang='fra'):
        salt = random.randint(32768, 65536)
        sign = make_md5(self.appid + query + str(salt) + self.appkey)

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'appid': self.appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

        # Send request
        r = requests.post(self.url, params=payload, headers=headers)
        result = dict(r.json())
        if 'error_code' in result.keys():
            return ""
        return result['trans_result'][0]['dst']


def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()


if __name__ == '__main__':
    translator = BaiduTextTranslator()
    en = ("Title: You Can Smell Hillary’s Fear. Body: Daniel Greenfield, a Shillman Journalism Fellow at the Freedom "
          "Center, is a New York writer focusing on radical Islam.")
    ru = translator.trans(en, from_lang='en', to_lang='fra')
    de = translator.trans(ru, from_lang='fra', to_lang='ru')
    back_en = translator.trans(de, from_lang='ru', to_lang='en')
    print("Original: " + en)
    print("En to Ru: " + ru)
    print("Ru to De: " + de)
    print("Back to En: " + back_en)
