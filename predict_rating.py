import numpy as np
import pandas as pd
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, SpatialDropout1D, GRU
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

business = pd.read_csv("yelp_final_project/yelp_filtered_business.csv")
review_all = pd.read_csv("yelp_final_project/yelp_filtered_review.csv")

a = business[business['categories'].str.contains('Restaurant') == True]
rev = review_all[review_all.business_id.isin(a['business_id']) == True]

rev_samp = rev.sample(n = 350000, random_state = 42)
train = rev_samp[0:280000]
test = rev_samp[280000:]

train = train[['text', 'stars']]
train['stars'].hist()
train.head()

train = pd.get_dummies(train, columns = ['stars'])
train.head()

test = test[['text', 'stars']]
test = pd.get_dummies(test, columns = ['stars'])
train.shape, test.shape

train_samp = train.sample(frac = .1, random_state = 42)
test_samp = test.sample(frac = .1, random_state = 42)

embed_size = 200 
max_features = 20000
maxlen = 200

embedding_file = 'yelp_final_project/glove.twitter.27B.200d.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embedding_file, encoding='utf8'))

class_names = ['stars_1', 'stars_2', 'stars_3', 'stars_4', 'stars_5']
y = train_samp[class_names].values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_samp['text'].values))
X_train = tokenizer.texts_to_sequences(train_samp['text'].values)
X_test = tokenizer.texts_to_sequences(test_samp['text'].values)
x_train = pad_sequences(X_train, maxlen = maxlen)
x_test = pad_sequences(X_test, maxlen = maxlen)

word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.zeros((nb_words, embed_size))
missed = []
for word, i in word_index.items():
    if i >= max_features: break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        missed.append(word)

inp = Input(shape = (maxlen,))
x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = True)(inp)
x = SpatialDropout1D(0.5)(x)
x = Bidirectional(LSTM(40, return_sequences=True))(x)
x = Bidirectional(GRU(40, return_sequences=True))(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
conc = concatenate([avg_pool, max_pool])
outp = Dense(5, activation = 'sigmoid')(conc)

model = Model(inputs = inp, outputs = outp)
earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3)
checkpoint = ModelCheckpoint(monitor = 'val_loss', save_best_only = True, filepath = 'yelp_lstm_gru_weights.hdf5')
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x_train, y, batch_size = 512, epochs = 20, validation_split = .1, callbacks=[earlystop, checkpoint])

y_test = model.predict([x_test], batch_size=1024, verbose = 1)

model.evaluate(x_test, test_samp[class_names].values, verbose = 1, batch_size=1024)

v = metrics.classification_report(np.argmax(test_samp[class_names].values, axis = 1),np.argmax(y_test, axis = 1))
print(v)
