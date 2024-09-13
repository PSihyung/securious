import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
from diffprivlib.mechanisms import Laplace

#1. 데이터 프레임생성
filename = 'C:/Users/user/OneDrive/문서/sandbox/주택담보대출승인장부.csv'
df = pd.read_csv(filename)


#2. 준식별자 범주화(연소득)
bins = [0,3000, 6000, 9000, 12000, float('inf')]
labels = ['3천만원 이하', '3천~6천', '6천~9천', '9천~1억2천', '1억2천 이상']
df['범주화_연소득'] = pd.cut(df['연소득'], bins=bins, labels=labels, right=False)
df['범주화_연소득'].value_counts()

#3 준식별자 범주화 (신용점수)
bins = [0, 600, 800, 900, 950, 1000]
labels = ['600 이하', '600~ 800', '800~ 900','900~950','900 ~ 1000']
df['범주화_신용점수'] = pd.cut(df['신용점수'], bins=bins, labels=labels, right=False)
df['범주화_신용점수'].value_counts()

#4. 준식별자 범주화(직장)
df['직업'].value_counts()

#5. 준식별자 그룹화 및 k익명성 구하기
quasi_identifiers = ['범주화_연소득','범주화_신용점수','직업']
df['group_id'] = df.groupby(quasi_identifiers).ngroup()
group_sizes = df.groupby('group_id').size().reset_index(name='group_size')

df = df.merge(group_sizes, on='group_id', how='left')
k_value = df['group_size'].min()
print(f"k익명성: {k_value}" )

#6. K익명성에 따른 정규화
k_counts = Counter(df['group_size'])
k_min = min(k_counts.keys())
k_max = max(k_counts.keys())

normalized_counts = {k : (k-k_min)/(k_max-k_min) for k, v in k_counts.items()}
#items함수는 딕셔너리의 키-값 쌍을 튜플형태의 리스트로 반환

#7. 정규화 값에 따른 그룹별 위험도 분류

risk_categories ={}

for k, v in normalized_counts.items():
    if v <= 0.2:
        risk_categories[k] = '매우높은위험'
    elif 0.2 < v < 0.4:
        risk_categories[k] = '높은위험'
    elif 0.4 < v < 0.6:
        risk_categories[k] = '위험'
    elif 0.6 < v < 0.8:
        risk_categories[k] = '낮은위험'
    else:
        risk_categories[k] = '매우낮은위험'

df['risk_cat'] = df['group_size'].map(risk_categories) #위험도 분류
df['risk'] = df['group_size'].map(normalized_counts) #위험도(정규화된 데이터)

#8. 위험도 별 엡실론 값 적용
def laplace_mechanism(value,epsilon,sensitivity):
  if epsilon ==0:
    epsilon = 0.1

  laplace_mechanism = Laplace(epsilon=epsilon,sensitivity=sensitivity)
  dp_value = laplace_mechanism.randomise(value)
  
  if dp_value < 0:
    dp_value = 0
  
  return dp_value

sensitivity = 1.0

#9. 연소득에 엡실론 추가

df['연소득'] = df.apply(lambda row: laplace_mechanism(row['연소득']/100,row['risk'],sensitivity),axis=1)
df['연소득'] = (df['연소득']*100).round().astype(int)

#10. 신용점수에 엡실론 추가

df['신용점수'] = df.apply(lambda row: laplace_mechanism(row['신용점수']/10,row['risk'],sensitivity),axis=1)
df['신용점수'] = (df['신용점수']*10).round().astype(int)
df['신용점수'] = df['신용점수'].apply(lambda x: min(x, 1000))

#11. DTI에 앱실론 추가

df['DTI'] = df.apply(lambda row: max(laplace_mechanism(row['DTI'],row['risk'], sensitivity), 1), axis=1)
df['DTI'] = df['DTI'].round().astype(int)


#12. 담보가치 에 앱실론 추가

df['담보가치'] = df.apply(lambda row: laplace_mechanism(row['담보가치']/100,row['risk'],sensitivity),axis=1)
df['담보가치'] = (df['담보가치']*100).round().astype(int)

#12. 근속연도에 앱실론 추가

df['근속연도'] = df.apply(lambda row: max(laplace_mechanism(row['근속연도'], row['risk'], sensitivity), 1), axis=1)
df['근속연도'] = df['근속연도'].round().astype(int)


subset_df2 = df[['직업','연소득', '신용점수', 'DTI','근속연도','담보가치','승인여부']]



subset_df2.to_csv('C:/Users/user/OneDrive/문서/sandbox/차등적용주택담보대출승인장부.csv', index=False, encoding='utf-8-sig')

