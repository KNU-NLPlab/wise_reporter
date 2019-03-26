# -*- coding:utf-8 -*-
#  summary_demo.py
# 18.11.15
# wrapper for 2차년도 report project demo

import os
import sys
sys.path.extend(['modules/multi_summ'])
import argparse
import collections
import json
import codecs

import multiprocessing as mp
from sklearn.feature_extraction.text import TfidfVectorizer

import translate
import morph_to_normal_convert as m_to_n_convert

from onmt.translate.Translator import make_translator

# fine tuning model
model_path = 'modules/multi_summ/dataset_m2s2/m2s_result_e26_acc_48.70_ppl_14.23.pt'

# model input temp 파일 저장 디렉토리
article_path = 'modules/multi_summ/dataset_m2s2/{}_model_input.txt'

# model output temp 파일 저장 디렉토리
res_path = 'modules/multi_summ/dataset_m2s2/{}_model_output.txt'

# model attention 표시용 json 파일 저장 디렉토리
json_path = 'results/{}/summary'


def make_tmp_opt(keyword):
    # 모델 option 설정 / gpu 아이디 설정 요망
    opt = argparse.Namespace(alpha=0.0, attn_debug=False, batch_size=1, beam_size=5, beta=-0.0, block_ngram_repeat=3,
                             coverage_penalty='none', data_type='text', dump_beam='', dynamic_dict=True, gpu=0,
                             ignore_when_blocking=[], length_penalty='none', max_length=400, max_sent_length=None,
                             min_length=0, model=model_path, n_best=1, output=res_path.format(keyword), replace_unk=True,
                             report_bleu=False, report_rouge=False, sample_rate=16000, share_vocab=False,
                             src=article_path.format(keyword), src_dir='', stepwise_penalty=False, tgt=None, verbose=False,
                             window='hamming', window_size=0.02, window_stride=0.01)
    return opt


opt = make_tmp_opt("")
translator_sum = make_translator(opt, report_score=True)


# 소주제별 tf_idf score 계산 함수
# stc_extract 함수에 의존
def tf_idf(list):
    vector = TfidfVectorizer()
    x = vector.fit_transform(list)
    result = []

    for j in range(len(x.toarray())):
        try:
            summ = 0
            temp = []
            for i in range(len(x.toarray()[j])):
                summ += x.toarray()[j][i]
            temp.append(str(summ))
            temp.append(str(list[j]))
            result.append(temp)
        except:
            pass

    return result


# 상위 n개 문장 선택 함수
# stc_extract 함수에 의존
# def top_n_stc(doc, n):
#     stc = ""
#     temp_list = []
#     for i in range(len(doc)):
#         temp_list.append(doc[i][1])
#     temp_list = uniq(temp_list)
#     print(len(doc))
#     print(len(temp_list))
#     for i in range(len(temp_list)):
#         if (i == int(n)):
#             break
#         else:
#             stc += str(temp_list[i]) + ' '
#     # print(str(len(str(stc).split(' '))) + '\n')
#     return stc

def top_n_stc(doc, n):
    stc = ""
    temp_list = []
    for i in range(len(doc)):
        temp_list.append(doc[i][1])
    temp_list = uniq(temp_list)
    #print(len(doc))
    #print(len(temp_list))
    for i in range(len(temp_list)):
        if (1100 <= len(stc.split(' '))):
            break
        else:
            stc += str(temp_list[i]) + ' '
    # print(str(len(str(stc).split(' '))) + '\n')
    return stc

def uniq(input):
    output = []
    for x in input:
        if x not in output:
            output.append(x)
    return output

# 이거 함수 추가
def all_docs(docs_list):
    result_docs = []
    for i in range(len(docs_list)):
        for j in range(len(docs_list[i])):
            result_docs.append(docs_list[i][j])
    return result_docs

# 함수 교체
def stc_extract(query, keywords, docs_stc):
#     print(keywords[0])
#     print(docs_stc[0])
    tf_idf_result_pair = []
    result_stc = []
    dic_feature_name = {}
    
    all_stc = all_docs(docs_stc)
    
    vector = TfidfVectorizer(token_pattern='[^\s+]+')
#     vector = TfidfVectorizer()
    x = vector.fit(all_stc)
    feature_names =x.get_feature_names()
    print('feature_names_length = ', len(feature_names))
    
#     query_key = query.split(' ')
#     for i in range(len(keywords)):
#         keywords[i] += query_key
#     print(keywords)
    
    for i in range(len(feature_names)):
        dic_feature_name[str(feature_names[i])] = i
    
    for i in range(len(docs_stc)):
        temp_pair = []
        for j in range(len(docs_stc[i])):
            trans_x = vector.transform([docs_stc[i][j]])
            trans_x_arr = trans_x.toarray()[0]
            summ_tf_idf = 0
            for k in range(len(keywords[i])):
                summ_tf_idf += trans_x_arr[dic_feature_name[str(keywords[i][k])]]
            temp_tf_idf = (summ_tf_idf, docs_stc[i][j])
            temp_pair.append(temp_tf_idf)
        tf_idf_result_pair.append(temp_pair)
#     print(tf_idf_result_pair[0])
    for i in range(len(tf_idf_result_pair)):
        tf_idf_result_pair[i] = sorted(tf_idf_result_pair[i], reverse=True)
        temp = top_n_stc(tf_idf_result_pair[i], 30)
        result_stc.append(temp)
        
    return result_stc


## tf_idf score 기반 상위 n개 만큼 문장 추출 함수
#def stc_extract(stc_list):
#    result_stc = []
#    temp_list = []
#    for i in range(len(stc_list)):
#        #print(stc_list[i])
#        temp_list.append(tf_idf(stc_list[i]))
#
#    for i in range(len(temp_list)):
#        temp_list[i] = sorted(temp_list[i], reverse=True)
#        temp_stc = top_n_stc(temp_list[i], 20)
#        result_stc.append(temp_stc)
#
#    return result_stc



def make_tmp_input(keyword, article_list):
    with open(article_path.format(keyword), 'w', encoding="utf-8") as tmp_file:
        print('\n'.join(article_list), file=tmp_file)


def read_result(keyword):
    with open(res_path.format(keyword), 'r', encoding="utf-8") as tmp_file:
        res = []
        for line in tmp_file:
            res.append(line.strip())
            #res = tmp_file.read().strip()
    return res


def make_demo_attn_info(article, gen_abstract, raw_attn_probs, output_file_path, num):
    num_class = 10

    def relatvie_normalizing(input_list):
        def check_class(range_list, value):
            for i, range in enumerate(range_list[:-1]):
                if value <= range:
                    return i
            return i + 1

        max_prob = max(input_list)
        min_prob = min(input_list)

        step = (max_prob - min_prob) / 10
        range_list = [min_prob + step * x for x in range(1, 11)]
        normalized_index = [check_class(range_list, value) for value in input_list]

        return normalized_index
    
    if os.path.exists(output_file_path) == False:
        os.makedirs(output_file_path)
    
    with open(os.path.join(str(output_file_path), "detail" + str(num) + '.json'), 'w', encoding="utf-8") as out_file:
        json_data = collections.OrderedDict()
        normalized_attn_probs = []

        json_data["article"] = article
        json_data["gen_abstract"] = gen_abstract
        json_data["raw_attn_probs"] = raw_attn_probs

        if len(json_data["raw_attn_probs"]) == len(json_data["gen_abstract"].split()) + 1:
            json_data["raw_attn_probs"] = json_data["raw_attn_probs"][:-1]

        for attns in json_data["raw_attn_probs"]:
            normalized_index = relatvie_normalizing(attns)
            normalized_attn_probs.append(normalized_index)

        assert len(normalized_attn_probs) == len(json_data["raw_attn_probs"])
        assert len(normalized_attn_probs[0]) == len(json_data["raw_attn_probs"][0])
        assert len(normalized_attn_probs[0]) == len(json_data["article"].split())
        assert len(json_data["gen_abstract"].split()) == len(normalized_attn_probs)

        json_data["normalized_attn_probs"] = normalized_attn_probs
        json.dump(json_data, out_file, ensure_ascii=False, indent=4)
        
    return json_data


def make_summary(queue, keyword, article_list):

    # 소주제별 추출 문장 출력
#    print("INPUT TEXT : " + str(morphemized_article))

    make_tmp_input(keyword, article_list)

    opt = make_tmp_opt(keyword)
    
    # 밑에 소스랑 갈아끼우면 됨
    attns_info, oov_info, copy_info, raw_attns_info = translate.main(opt)

    # 2018/11/22
    # GPU memroy flush
    #print("paraent pid", os.getpid())
    #ctx = mp.get_context('spawn')
    #queue = ctx.Queue()
    #p = ctx.Process(target=translate.sub_main, args=(queue, opt))
    #p.start()

    #p.join(3) # timeout 안 설정하면 안끝남

    #attns_info, oov_info, copy_info, raw_attns_info = queue.get()
#    print("exit sub process")
#

    res = read_result(keyword)

    # 입력에 대한 모델 결과(요약 출력)
#    print("OUTPUT TEXT : " + str(res))

    json_list = []
    for i in range(len(raw_attns_info)):
        json_obj = make_demo_attn_info(article_list[i], res[i], raw_attns_info[i], json_path.format(keyword), i)
        json_list.append(json_obj)

    gen_abs_list = [ dic["gen_abstract"] for dic in json_list ]
    normal_word_gen_abs_list = m_to_n_convert.convert(gen_abs_list, keyword)

#    print("DEMO JSON : " + str(json_obj))
    queue.put((normal_word_gen_abs_list, json_list))
    
    
def make_summary_preloaded(keyword, article_list):
    make_tmp_input(keyword, article_list)
    opt = make_tmp_opt(keyword)
    translator_sum.out_file = codecs.open(opt.output, 'w', 'utf-8')
    _, attns_info, oov_info, copy_info, raw_attns_info = translator_sum.translate(opt.src_dir, opt.src, opt.tgt,
                         opt.batch_size, opt.attn_debug, raw_attn=True)
    
    res = read_result(keyword)
    json_list = []
    for i in range(len(raw_attns_info)):
        json_obj = make_demo_attn_info(article_list[i], res[i], raw_attns_info[i], json_path.format(keyword), i)
        json_list.append(json_obj)
        
    gen_abs_list = [ dic["gen_abstract"] for dic in json_list ]
    return m_to_n_convert.convert(gen_abs_list)


# 입력 : 소주제별 문장 list
# 출력 : demo_json (demo용 json list 리턴 / 소주제 갯수만큼 존재)
#       demo_result (demo용 model output 리턴 / 소주제 갯수만큼 존재)
# etc : 함수명 변경 요망
def main(list):

    # 변수명 : demo_json
    # 내용 : detail json (attention 표시용 json)
    # type : list
    demo_json = []

    # 변수명 : demo_result
    # 내용 : result (요약 결과)
    # type : list
    demo_result = []

    prepro = stc_extract(list)

    for i in range(len(prepro)):
        demo_json.append(make_summary(prepro[i], i))

    for i in range(len(demo_json)):
        demo_result.append(demo_json[i]['gen_abstract'])

    return demo_json, demo_result


if __name__ == "__main__":

    # sample input list
    list = [
        ['이건 저건', '저건 요건', '요건 김건', '김건 길건', '호로 로 롤롤 로 로 로 롤 로로', '호잇 호잇 호호 호잇', ],
        ['비비 삠삠 삐 삐 비비 비', '바리바리 바바 바리 리리 바리', '큐큐 큐 후룰루 랄라라리라', '혼또 혼또 혼혼혼'],
        ['비비 삠삠 삐', '쿠르릅 쿠아아아', '호잇 호호 보보보보', '쿠왑 빠압 삡']
    ]

    main(list)
