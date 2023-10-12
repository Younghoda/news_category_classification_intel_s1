# 자연어를 숫자로 바꿔주기 -> 숫자에 의미를 줘야한다
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split    # pip install scikit-learn 설치
from konlpy.tag import Okt  #pip install konlpy 설치
from tensorflow.keras.preprocessing.text import Tokenizer   #pip install tensorflow==2.7.0
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

pd.set_option('display.unicode.east_asian_width', True) #titles  category와 내용 자리 맞춰주기 위해
df = pd.read_csv('./crawling_data/naver_all_news.csv')
print(df.head())
df.info()

X = df['title']
Y = df['category']

encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)    # 라벨로 바꿔줌
print(labeled_y[:3])    # [3 3 3]
label = encoder.classes_
print(label)
with open('./models/encoder.pickel', 'wb') as f:
    pickle.dump(encoder, f)

onehot_y = to_categorical(labeled_y)
# print(onehot_y) #  [0. 0. 0. 1. 0. 0.]

okt = Okt() # konlpy -> X축을 자연어 처리하기 위해 -> 한글을 자연어 처리할려면 KoNLPy(코엔엘파이) ->
# [국정원, 사전투표, 여부, 조작, 땐, 본, 투표, 서, 이중, 투표, 가능]를 각각의 단어(형태소)를 숫자화하는 것이 토크라이저라고 한다.
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)   # stem=True롤 사용하면 동사 원형으로 가져옴 -> 데이터가 많으면 생략 가능
# print(X)
# 한 글자짜리는 의미가 없는 경우가 많기 때문에 지워준다. == stop word(불용어)제거

stopwords = pd.read_csv('./stopwords.csv', index_col=0)
for j in range(len(X)): #
    words =[]
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:    # 길이가 1보다 크면
            if X[j][i] not in list(stopwords['stopword']):  # stopword에 없으면(불용어가 아니면)
                words.append(X[j][i])   #추가한다.
    X[j] = ' '.join(words)   #' '를 이용해서 붙여준다. -> 리스트를 문장으로 만들어줌
# print(X[0])

token = Tokenizer() # 라벨링하는 것 -> 형태소하나하나에 라벨로 바꿔줌
token.fit_on_texts(X)   # 형태소 하나하나에 라벨로 바꿔줌
tokened_x = token.texts_to_sequences(X) # 형태소를 숫자로 바꿔서 라벨 리스트를 만들어 준다.
wordsize = len(token.word_index) + 1   # 유니크한 번호의 갯수(우리는 0을 사용할 것이기 때문에 +1 해줌)
print(tokened_x[0:3])
print(wordsize)
# 피클 사용(그대로 저장하기 위해) -> encoder와 Tokenizer를 저장시켜줘야함
with open('./models/news_token.pickle', 'wb') as f:
    pickle.dump(token, f)

max = 0
for i in range(len(tokened_x)):
    if max < len(tokened_x[i]):
        max = len(tokened_x[i])
print(max)

x_pad = pad_sequences(tokened_x, max)   # 모자라는 개수만큼 0을 넣어서 숫자를 맞춤
print(x_pad[:3])

X_train, X_test, Y_train, Y_test = train_test_split(
    x_pad, onehot_y, test_size=0.2)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test
np.save('./crawling_data/news_data_max_{}_wordsize_{}'.format(max, wordsize), xy)



