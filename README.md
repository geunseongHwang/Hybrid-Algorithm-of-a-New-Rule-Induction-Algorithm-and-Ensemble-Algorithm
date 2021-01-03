(Master's research)
Hybrid Algorithm of a New Rule Induction Algorithm and Ensemble Algorithm
- Rule Induction과 Ensemble의 Hybrid Model   -
	- Rule_test: 하이브리드 모델, 의사결정나무모델, 랜덤포레스트, 그래디언트부스팅을 특정데이터로 부터 해석력 또는 성능을 비교하기 위한 코드
	- rule_attr_change: 생성된 rule set의 분기기준을 형태에 맞게 재설정 하기 위한 코드
	- comparison: 각 주어진 모델별 전체 데이터의 반복한 수치를 평균 또는 중위수로 뽑기 위한 코드
	- model_info: 각 모델별 해석력과 관련된 수치를 누적 샘플커버리지를 기준으로 비교하기 위한 코드
	- excel_creation: 리스트로 저장된 데이터프레임들을 한 시트의 엑셀로 뽑기위한 코드
	- rulebase: One sieded max purity를 기준으로 뽑은 규칙 유도 모델과 더불어 규칙 유도 모델을 통한 클래스 예측을 하기 위한 코드
	- rulebase_excel: One sieded max purity를 기준으로 뽑은 규칙 유도 모델과 더불어 규칙 유도 모델을 통한 클래스 예측을 하기 위한 코드(+ Decision Tree 그래프 생성시 필요)
	- splitcriterion: 분기기준을 정하기 위한 코드
	- usertree: 정해진 분기기준을 통해 의사결정나무를 생성하기 위한 코드
	- utils: 의사결정나무 그래프 생성시 필요한 데이터를 뽑거나 실제값과 예측값 성능 등을 보기 위한 코드
	- visgraph: 동질성, 샘플커버리지, 예측 등 DT 그래프 생성시 필요한 코드
	- (Gaussian)Concise_Rule: (정규분포)가설검정을 통해 규칙들마다 독립성을 보고 규칙의 분기기준 수를 재정립하기 위한 코드
  	- (Chi-square)Concise_Rule: (카이제곱분포)가설검정을 통해 규칙들마다 독립성을 보고 규칙의 분기기준 수를 재정립하기 위한 코드
