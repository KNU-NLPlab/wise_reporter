# 어떤 주제에 대한 빅데이터를 스마트 보고서로 요약하는 기술 개발
'어떤 주제에 대한 빅데이터를 스마트 보고서로 요약하는 기술 개발'의 통합 관리 Repository입니다.

## 모듈 현황 및 Refactoring 진행 상황
1. Topic Generator
2. Multi Document Summarizer (modules/multi_summ/dataset_m2s2/[Model](https://drive.google.com/file/d/1YEqkteTMnoQhZBeUm6fbbEew8LH_xyeF/view?usp=sharing))
3. Timeline Summarizer (modules/timeline_summ/model/[Model](https://drive.google.com/open?id=1ylL4InDMU6EZ6hCv-w8C0IsvSH3DuOLf))
4. Image Selector
5. Outlook (modules/forecast/model/[Model](https://drive.google.com/open?id=1DkHGDm2F3uSgCuTNnHVjWKTJpALDNhts))

## 구현 가이드라인

modules의 base_module.py를 상속받아서 abstract method를 구현해주셔야 합니다.

내부적으로 처리하는 inner process의 경우에는 각자 팀 별 폴더에서 관리해주시면 됩니다.

modules의 이미 구현된 다른 모듈(1, 2번 Module)들을 참고하시면 됩니다.

## 문의사항

최우용 석사과정 (wychoi@sejong.knu.ac.kr) 에게 문의바랍니다.
