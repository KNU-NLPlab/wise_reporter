# wise_reporter

## 임시
[test.ipynb](https://github.com/KNU-NLPlab/wise_reporter/blob/master/test.ipynb) 참조

## 모듈 현황 및 Refactoring 진행 상황
1. Document Graph Analyzer
2. Multi Document Summarizer  
    Model: https://drive.google.com/file/d/1YEqkteTMnoQhZBeUm6fbbEew8LH_xyeF/view?usp=sharing
3. Timeline Summarizer  
    Model: https://drive.google.com/open?id=1jC_Ygf2C_D3aOoF78gIIM9W1BmNUbZZD  
    Path: modules/timeline_summ/model/  
4. ~~Image Retriever~~
5. ~~Image Caption Generater~~
6. Outlook  
    Model: https://drive.google.com/open?id=1DkHGDm2F3uSgCuTNnHVjWKTJpALDNhts  
    Path: modules/forecast/model/  

## 구현 가이드라인

modules의 base_module.py를 상속받아서 abstract method를 구현해주셔야 합니다.

내부적으로 처리하는 inner process의 경우에는 각자 팀 별 폴더에서 관리해주시면 됩니다.

modules의 이미 구현된 다른 모듈(1, 2번 Module)들을 참고하시면 됩니다.

## 문의사항

최우용 석사과정 (wychoi@sejong.knu.ac.kr) 에게 문의바랍니다.

다중문서요약 모델 관련해서는 신용민 석사과정 (ymshin@sejong.knu.ac.kr) 에게 문의바랍니다.

