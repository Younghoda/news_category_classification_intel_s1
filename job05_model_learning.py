import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

X_train, X_test, Y_train, Y_test = np.load(
    './crawling_data/news_data_max_20_wordsize_11995.npy', allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(11995, 300, input_length=20))  # Embedding: 단어 개수 만큼의 차원을 갖을 수 있도록 하고 벡터값으로 바꿔준다. -> 의미 공간상의 벡터화(좌표화) ,
# 11995차원을 300차원으로 축소(차원이 커질수록 데이터의 개수는 그대로이기 때문에 의미가 없어짐 이것을 막기 위해 차원을 축소화 한다.)
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu')) # 1D: 문장은 한줄이기 때문에, 필터 32개
model.add(MaxPooling1D(pool_size=1))    # 1이기 때문에 아무일도 일어나지 않는다.
model.add(LSTM(128, activation='tanh', return_sequences=True))  # return_sequences=True이게 없으면 맨 마지막 데이터만 저장
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6,activation='softmax'))    # 출력은 6개, 카테고리 분류는 softmax
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
model.save('./models/news_category_classification_model_{}.h5'.format(fit_hist.history['val_accuracy'][-1]))
plt.plot(fit_hist.history['val_accuracy'], label='validation accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()
