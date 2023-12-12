#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 특이값 분해를 위한 TruncatedSVD import
from sklearn.decomposition import TruncatedSVD
# 특이값 분해를 위한 svds import
from scipy.sparse.linalg import svds

# 데이터를 차트나 플롯으로 그려주는 라이브러리 패키지인 pyplot import
import matplotlib.pyplot as plt
# 통계 그래프를 그려주는 라이브러리 패키지인 seaborn import
import seaborn as sns
# 데이터 프레임과 시리즈를 사용하기 쉽게 해주는 라이브러리 패키지인 pandas import
import pandas as pd
# 수학적 연산을 쉽게 해주는 라이브러리 패키지인 numpy import
import numpy as np
# 경고 메시지를 무시하기 
import warnings
warnings.filterwarnings("ignore")

# In[3]:


# ** 특정 장소와 유사한 장소 가져오는 알고리즘 ** 

# 장소 csv 데이터 가져오기
place_data = pd.read_csv('PetPlace.csv', encoding='cp949')
# 유저 csv 데이터 가져오기  
users_data = pd.read_csv('User0419 (3).csv', encoding='cp949')

# In[4]:


# 장소 데이터 정보 확인하기 
place_data.head()

# In[5]:


# 유저 데이터 정보 확인하기 
users_data.head()

# In[6]:


# 장소 데이터 모양 출력
print(place_data.shape)
# 유저 데이터 모양 출력 
print(users_data.shape)

# In[7]:


# 데이터 전처리 - 필요하지 않은 데이터들 지우기
place_data.drop('지번주소', axis = 1, inplace = True)
place_data.drop('전화번호', axis = 1, inplace = True)
place_data.drop('휴무일', axis = 1, inplace = True)
place_data.drop('운영시간', axis = 1, inplace = True)
place_data.drop('최종작성일', axis = 1, inplace = True)
place_data.drop('주차 가능여부', axis = 1, inplace = True)

# 전처리된 장소 데이터 정보 확인하기 
place_data.head()

# In[8]:


# '매장 고유 번호' 를 기준으로 사용자 데이터와 장소 데이터 병합
user_place_data = pd.merge(users_data, place_data, on = '매장 고유 번호')
# 병합된 데이터 정보 확인하기
user_place_data.head()

# In[9]:


# 병합된 데이터 모양 출력
user_place_data.shape

# In[10]:


# cloumn은 시설명, row는 사용자 번호, value는 평점인 pivot table로 데이터 변경
# 즉, 데이터가 사용자-장소 평점 데이터로 변경됨
# 사용자가 평점을 매기지 않은 정보에 들어가는 NaN 값을 fillna를 사용해 0으로 변경
user_place_rating = user_place_data.pivot_table('평점', index = '사용자 번호', columns = '시설명').fillna(0)

# In[11]:


# pivot table로 바꾼 데이터 모양 출력
user_place_rating.shape

# In[12]:


# pivot table로 바꾼 데이터 정보 확인하기 
user_place_rating.head()

# In[13]:


# 사용자-장소로 되어 있는 기존 데이터를 numpy.T (Transpose) 를 이용하여 전치
# 즉, 장소-사용자 데이터로 변경
place_user_rating = user_place_rating.values.T
# 장소-사용자로 바꾼 데이터 모양 출력 
place_user_rating.shape

# In[14]:


# 장소-사용자 데이터 타입 확인 
type(place_user_rating)

# In[15]:


# scikit learn의 TruncatedSVD를 사용하여 특이값 분해(SVD) 함.
# latent 값은 12로 지정. 이 경우 변환된 데이터가 12개 요소의 값을 갖게 됨.
# 장소-사용자 데이터를 2개의 주요 component로 TruncatedSVD 변환
SVD = TruncatedSVD(n_components=12)
matrix = SVD.fit_transform(place_user_rating)
# 행렬의 모양 출력 
matrix.shape

# In[16]:


# 행렬에 들어간 12개 요소의 값 확인
matrix[0]

# In[17]:


# numpy의 corrcoef를 이용하여 피어슨 상관계수를 구함.
corr = np.corrcoef(matrix)
# 구해진 상관계수 모양 확인
corr.shape

# In[18]:


# corr의 데이터 200번째 까지의 요소 꺼내기
corr2 = corr[:200, :200]
# 바뀐 데이터 모양 확인
corr2.shape

# In[19]:


# matplotlib를 사용하여 그래프 출력
# 가로길이 16, 세로길이 10 짜리의 그래프 제작
plt.figure(figsize=(16, 10))
# Seaborn을 이용해 corr2 값의 heatmap 제작
sns.heatmap(corr2)

# In[20]:


# '특정 장소'와 관련하여 상관계수가 높은 장소 뽑기
# 장소 이름 = 사용자-장소 데이터의 열
place_title = user_place_rating.columns
# 장소 이름 목록 = 장소 이름을 리스트화
place_title_list = list(place_title)
# "2Cats" 를 특정 장소로 지정
coffey_hands = place_title_list.index("2Cats")

# In[21]:


# "2Cats"의 상관계수를 가져옴
corr_coffey_hands  = corr[coffey_hands]
# "2Cats"의 상관계수와 95% 유사한 장소 상위 50개를 리스트화 하여 출력 
list(place_title[(corr_coffey_hands >= 0.95)])[:50]

# In[29]:


# ** 특정 사용자와 유사한 사용자의 선호 장소를 가져오는 알고리즘 **

# 장소 csv 데이터 가져오기
df_places = pd.read_csv('PetPlace.csv', encoding='cp949')
# 사용자 csv 데이터 가져오기
df_ratings = pd.read_csv('User0419 (3).csv', encoding='cp949')

# In[30]:


# cloumn은 매장 고유 번호, row는 사용자 번호, value는 평점인 pivot table로 데이터 변경
# 사용자가 평점을 매기지 않은 정보에 들어가는 NaN 값을 fillna를 사용해 0으로 변경
df_user_place_ratings = df_ratings.pivot(
    index='사용자 번호',
    columns='매장 고유 번호',
    values='평점'
).fillna(0)

# In[31]:


# 사용자-데이터 pivot table 데이터 출력 
df_user_place_ratings.head()

# In[32]:


# matrix는 pivot_table 값을 numpy matrix로 만든 것 
matrix = df_user_place_ratings.to_numpy()

# user_ratings_mean은 사용자의 평균 평점 
user_ratings_mean = np.mean(matrix, axis = 1)

# matrix_user_mean : 사용자-장소에 대해 사용자 평균 평점을 뺀 것.
matrix_user_mean = matrix - user_ratings_mean.reshape(-1, 1)

# In[33]:


# matrix 출력
matrix

# In[34]:


# 행렬 모양 확인
matrix.shape

# In[35]:


# 사용자의 평균 평점 모양 확인 
user_ratings_mean.shape

# In[36]:


# matrix_user_mean 모양 확인 
matrix_user_mean.shape

# In[37]:


# Pandas를 이용하여 DataFrame 출력
pd.DataFrame(matrix_user_mean, columns = df_user_place_ratings.columns).head()

# In[38]:


# scipy에서 제공해주는 svd 사용
# svds는 TruncatedSVD 개념을 사용 
# U 행렬, sigma 행렬, V 전치 행렬을 반환.

U, sigma, Vt = svds(matrix_user_mean, k = 12)

# In[39]:


# 각 행렬의 모양 확인 
print(U.shape)
print(sigma.shape)
print(Vt.shape)

# In[40]:


# 0이 포함된 대칭행렬로 변환하기 위해 numpy의 diag 이용
sigma = np.diag(sigma)

# In[41]:


# 데이터 모양 확인
sigma.shape

# In[42]:


# 데이터 인덱스 0의 값 출력
sigma[0]

# In[43]:


# 데이터 인덱스 1의 값 출력
sigma[1]

# In[44]:


# 데이터 matrix_user_mean은 SVD를 적용하여 분해된 상태
# U, Sigma, Vt의 내적을 수행하여 다시 원본 행렬로 복원
# 거기에 아까 뺐던 사용자 평균 rating을 더함
svd_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

# In[45]:


# Pandas를 이용하여 DataFrame 만듦
df_svd_preds = pd.DataFrame(svd_user_predicted_ratings, columns = df_user_place_ratings.columns)
# 만든 DataFrame 출력
df_svd_preds.head()

# In[46]:


# 데이터 모양 확인 
df_svd_preds.shape

# In[54]:


# 추천 함수 제작
# 함수의 인자로 사용자 아이디, 장소 정보 테이블, 평점 테이블을 받음
# 사용자 아이디에 SVD로 나온 결과의 평점이 가장 높은 데이터 순으로 정렬
# 사용자가 평가한 데이터는 제외하고, 평가하지 않은 데이터 중 평점이 높은 장소 추천
def recommend_places(df_svd_preds, user_id, ori_places_df, ori_ratings_df, num_recommendations=5):
    
    #현재는 index로 적용이 되어있으므로 user_id - 1을 해야함.
    user_row_number = user_id - 1 
    
    # 최종적으로 만든 pred_df에서 사용자 index에 따라 장소 데이터 정렬 -> 장소 평점이 높은 순으로 정렬 됌
    sorted_user_predictions = df_svd_preds.iloc[user_row_number].sort_values(ascending=False)
    
    # 원본 평점 데이터에서 user id에 해당하는 데이터를 뽑아낸다. 
    user_data = ori_ratings_df[ori_ratings_df['사용자 번호'] == user_id]
    
    # 위에서 뽑은 user_data와 원본 장소 데이터를 합친다. 
    user_history = user_data.merge(ori_places_df, on = '매장 고유 번호').sort_values(['평점'], ascending=False)
    
    # 원본 장소 데이터에서 사용자가 본 장소 데이터를 제외한 데이터를 추출
    recommendations = ori_places_df[~ori_places_df['매장 고유 번호'].isin(user_history['매장 고유 번호'])]
    # 사용자의 장소 평점이 높은 순으로 정렬된 데이터와 위 recommendations을 합친다. 
    recommendations = recommendations.merge( pd.DataFrame(sorted_user_predictions).reset_index(), on = '매장 고유 번호')
    # 컬럼 이름 바꾸고 정렬해서 return
    recommendations = recommendations.rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :]
                      
    # 사용자 데이터와 추천 장소 반환
    return user_history, recommendations

# In[55]:


# 206번 사용자가 Matrix Factorization 기반으로 추천받은 장소 가져오기
already_rated, predictions = recommend_places(df_svd_preds, 10 ,df_places, df_ratings, 10)

# In[56]:


# 상위 10개 데이터 출력 
already_rated.head(10)

# In[57]:


# 유사도 출력 
predictions

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



