#!/usr/bin/env python
# -*- coding : utf-8 -*-
# 18.06.21
# emtailment_scorer.py
# entailment reward model for deep summarization
import timeit
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from allennlp.common.checks import check_for_gpu
from allennlp.models.archival import load_archive
# from allennlp.predictors import Predictor
from allennlp.service.predictors import Predictor



class Entailment_scorer:
  def __init__(self, model_path, entail_index, cuda_device=0):
    self.entail_index = entail_index # index of entailment
    self.predictor = self.get_predictor(model_path, cuda_device)

  def get_predictor(self, model_path, cuda_device):
    check_for_gpu(cuda_device)
    archive = load_archive(model_path, weights_file=None, cuda_device=cuda_device, overrides="")

    return Predictor.from_archive(archive, None)

  def predict_entailment(self, json_data):
    results = self.predictor.predict_json(json_data)

    entail_prob = results["label_probs"][self.entail_index]

    return entail_prob

  def predict_batch_entailment(self, premise_generator, hyp_generator):
    batch_json_data = []
    entail_probs = []

    def update_result(prob_list, results):
      for result in results:
        #print(result)
        prob_list.append(result["label_probs"][self.entail_index])

    for premise, hyp in zip(premise_generator, hyp_generator):
      premise = premise.strip()
      hyp = hyp.strip()
      
      json_data = {"premise":premise, "hypothesis":hyp}
      batch_json_data.append(json_data) 
      if len(batch_json_data) == self.batch_size:
        results = self.predictor.predict_batch_json(batch_json_data)
        update_result(entail_probs, results)

        batch_json_data = []
        results = None
    if len(batch_json_data) != 0:
      results = self.predictor.predict_batch_json(batch_json_data)
      update_result(entail_probs, results)

      batch_json_data = []
      results = None
    
    return entail_probs  
    

if __name__ == "__main__":
  start = timeit.default_timer()
  model_path = "../entail/0617_model_400k/model.tar.gz"
  scorer = Entailment_scorer(model_path, 1)
  json_data = {"premise": "[ 싱가포르 = 이데일리 원다연 기자 ] 김정 은 북한 국무 위원 장 과 도널드 트럼프 대통령 이 12 일 오전 단독 회담 뒤 곧바로 확대 회담 에 돌입 하 었 다 . 확대 회담 에 는 미국 측 에서 마이크 폼페이오 국무 장관 과 존 볼턴 백악관 국가 안보 보좌 관 이 배석 하 었 다 . 북 측 에서 는 김영철 · 리수용 노동 당 부 위원 장 과 리용호 외무 상 이 배석 하 었 다 . 트럼프 대통령 은 확대 회담 모두 발언 을 통하 어 “ 김정 은 위원 장 과 크 ㄴ 문제 , 크 ㄴ 딜레마 를 해결 하 ㄹ 것 ” 이 라며 “ 함께 협력 하 어서 해결 하 어 나가 ㄹ 것 ” 이 이 라고 밝히 었 다 .", "hypothesis": "도널드 트럼프 미국 대통령 이 12 일 ( 한국 시간 ) 확대 회담 을 하 었 다 ."}

  start = timeit.default_timer()

  score = scorer.predict_entailment(json_data)
  print("score : {}".format(score))
  

  end = timeit.default_timer()
  print("Execution time: {}".format(end-start))
