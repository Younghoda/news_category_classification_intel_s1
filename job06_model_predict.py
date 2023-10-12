import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split    # pip install scikit-learn 설치
from konlpy.tag import Okt  #pip install konlpy 설치
from tensorflow.keras.preprocessing.text import Tokenizer   #pip install tensorflow==2.7.0
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.models import load_model

df = pd.read_csv('./crawling_data/naver_headline_news_20231012.csv')
print(df.head())
df.info()

X = df['titles']
Y = df['category']

with open('./models/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)
labeled_y = encoder.transform(Y)    # fit하면 정보를 새로 가져오게됨. 없으면 그대로 사용
label = encoder.classes_    # 라벨 정보

onehot_y = to_categorical(labeled_y)
print(onehot_y)

okt = Okt()

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
stopwords = pd.read_csv('./stopwords.csv', index_col=0)

for j in range(len(X)): # 전처리하는 코드 (불용어 제거)
    words =[]
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:    # 길이가 1보다 크면
            if X[j][i] not in list(stopwords['stopword']):  # stopword에 없으면(불용어가 아니면)
                words.append(X[j][i])   #추가한다.
    X[j] = ' '.join(words)   #' '를 이용해서 붙여준다. -> 리스트를 문장으로 만들어줌

with open('./models/news_token.pickle', 'rb') as f:
    token = pickle.load(f)

tokened_x = token.texts_to_sequences(X)
for i in range(len(tokened_x)):
    if len(tokened_x[i]) > 20:
        tokened_x[i] = tokened_x[i][:21]
x_pad = pad_sequences(tokened_x, 20)

model = load_model('./models/news_category_classification_model_0.7251387238502502.h5')
preds = model.predict(x_pad)
predicts = []
for pred in preds:
    most = label[np.argmax(pred)]
    pred[np.argmax(pred)] = 0
    second = label[np.argmax(pred)]
    predicts.append([most, second])
df['predict'] = predicts    # predict 안에 첫번재 두번째 값이 있다.
print(df.head(30))

df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'category'] in df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 'O'   #맞으면 O
    else:
        df.loc[i, 'OX'] = 'X'   # 틀리면 X
print(df['OX'].value_counts())
print(df['OX'].value_counts()/len(df))
for i in range(len(df)):
    if df['category'][i] not in df['predict'][i]:
        print(df.iloc[i])   # 틀린 것들만 확인하기 위해 프린트


