# KLUE_project
## train.py
각 모델을 학습시키는 코드입니다.
## inference.py
test 데이타를 인풋으로 학습된 모델의 출력을 받는 코드입니다.
## load_data.py
데이터를 불러오는 기능과 데이터 셋의 구현이 있는 코드입니다.
## evaluation.py
추론 결과를 평가하는 코드입니다.
## Notebook
데이터의 분석 및 데이터를 나누는 코드가 구현되어 있습니다.
### divide_data.ipynb
데이터의 분포를 분석하고 이 분포를 유지한채 validation set과 train set을 나누는 코드가 구현되어 있습니다.
### analyze_inference.ipynb
validation set의 추론 결과를 분석합니다. 각 레이블의 정답률과 개수의 상관관계를 분석합니다.
