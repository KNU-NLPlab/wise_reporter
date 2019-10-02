# -*- coding : utf-8 -*-
# etri_nlp.py
# It contain information to run etri nlp

# 18.01.02
# modify code for ETRI open api
# http://aiopen.etri.re.kr/doc_language.php
import json
import timeit
import pprint

from pip._vendor import urllib3

try:
    from python_etri_nlp.etri_client import Etri_client
except ImportError:
    from etri_client import Etri_client

class Etri_nlp():
    def __init__(self, host_addr = "155.230.90.217", port_number = 5004, silence = False):
        # etri nlp server address
        self.host_addr = host_addr
        # port number
        self.port_number = port_number
        self.client = Etri_client(self.host_addr, self.port_number, silence)

        # open API version
        self.openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"
        self.accessKey = "" # your key
        self.analysisCode = "ner" # 형태소 분석 : morp
                                    # 어휘의미 분석(동음이의어 분석) : "wsd"
                                    # 어휘의미 분석 (다의어 분석) : "wsd_poly"
                                    # 개체명 인식 : "ner"
                                    # 의존 구문 분석 : "dparse"
                                    # 의미역 인식 : "srl"
        self.http = urllib3.PoolManager()

    def get_parsed_json_from_api(self, sentence):
        requestJson = {
            "access_key": self.accessKey,
            "argument": {
                "text": sentence,
                "analysis_code": self.analysisCode
            }
        }
        response = self.http.request(
            "POST",
            self.openApiURL,
            headers={"Content-Type": "application/json; charset=UTF-8"},
            body=json.dumps(requestJson)
        )
        return json.loads(str(response.data, "utf-8"))['return_object']

    def get_parsed_json(self, sentence, error=0):
        # error : 너무 빠른 통신으로 인해 데이터를 다 받지 못하고 decode하는 걸 방지하기 위해
        #         sleep에 weight를 줌

        # connect to server
        self.client.connect()
        # send msg
        self.client.send(sentence)
        # recv result
        parsed_sentence = self.client.recv(error)
        self.client.close_connection()
        # print(parsed_sentence)
        return json.loads(parsed_sentence)

    def make_morp_sentence(self, json_dict):
        morp_element_list = []
        for sentence in json_dict['sentence']:
            # get morp
            morp_elements = [morp_info['lemma'] for morp_info in sentence['morp']]
            morp_element_list.extend(morp_elements)
            # save to file
        return ' '.join(morp_element_list)

    def single_sent_morph(self, json_dict) :
        return json_dict['sentence'][0]['morp_eval']

    # 17.12.02
    def get_noun_word(self, json_dict):
        noun_tag = {"NNG":0,"NNP":0}
        morp_element_list = []
        for sentence in json_dict['sentence']:
            # get morp
            morp_elements = [morp_info['lemma'] for morp_info in sentence['morp'] if morp_info['type'] in noun_tag]
            morp_element_list.extend(morp_elements)
            # save to file
        return morp_element_list

def get_word_morph(string) :
    etri_nlp = Etri_nlp(silence=True)
    json_dict = etri_nlp.get_parsed_json(string)
    return etri_nlp.single_sent_morph(json_dict)

def get_morph_sentence(string):
    etri_nlp = Etri_nlp(silence=True)
    json_dict = etri_nlp.get_parsed_json(string)
    return etri_nlp.make_morp_sentence(json_dict)

