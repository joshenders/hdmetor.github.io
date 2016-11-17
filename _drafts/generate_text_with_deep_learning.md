---
layout: post
title: Generate a new job posting à la Hacker News with Keras
description: "Use deeplearning library keras to generate a new job posting HacerkNews style"
modified: 2016-11-07
tags: [python, keras, deep learning, hacker news]
image:
  feature: abstract-4.jpg
  credit: dargadgetz
  creditlink: http://www.dargadgetz.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/
---
In this blog, we are going to show how to train a charachter-level deep learning model to hallucinate new jobs posting.

In order to train the model, we need data from the previous Hacker news posting. In order to do that, we will do a match query all query and grab the text field from the source.


```python
from elasticsearch import Elasticsearch
from elasticsearch import helpers

es = Elasticsearch()

body= {
    "query" : {
        "match_all" : {}
    }
}

res = list(helpers.scan(es,  query=body))
all_listings = [d['_source']['text'] for d in res if 'text' in d['_source']]

```

Note that we used the helper function `scan` (rather than `es.search`) because of the potentially large output.

## Organizing the text and counting the characters

Since we want to train a character level network, we need create a mapping from character to integers. Training the network on _all_ the character will be way too wasteful (if possible at all), and produce very noisy results. Therefore we are going to perform the following steps:

- use only one type of apostrophe

- lowercase all the text

- define a subset of `bad_chars` which will contain the least used characters (note that the threshold for that was selected manually after printing `counter.most_common()`)

- define the `good_chars` (the ones that will actually go into the model) as the complementary set to `bad_chars`

- pick a special `start_char`, `end_char` and `unknown_char` after making sure they don't belong in the `good_char` set

The reason why we need such special character is that we want to be able to generate a full listing, so we need to teach the model when a listing starts and ends.

```python
from collections import Counter

joined_listing = "".join(all_listings)
counter = Counter(joined_listing.lower().replace("\"", "'").replace("’", "'"))
chars = set(joined_listing)

bad_chars = [c for c, v in counter.most_common() if v < 2000] + ['—', '•']
good_chars = list(set(counter) - set(bad_chars))

start_char = '\x02'
end_char = '\x03'
unknown_char = '\x04'

# we don't want to pick characters that are already used
assert start_char not in good_chars
assert end_char not in good_chars
assert unknown_char not in good_chars

good_chars.extend([start_token, end_token, unknown_char])
```

We can now create the mapping from character to index (and vice versa, which will be useful at text generation time).


```python
char_to_int = {ch: i for i, ch in enumerate(good_chars)}
int_to_char = {i: ch for i, ch in enumerate(good_chars)}
```
# Tensorizing the text

It is now time to transform a list of strings (i.e. a list that where each element is a different posting) to a 3 dimensional tensor.

We know that the input has to have shape `(num_timestamps, seq_len, num_chars)`, where `seq_len` is an arbitrary number. It does represent the length of the sequence learn by the model.

## Step size

The number of timestamps (i.e. number of different training sequences) depends on the length of the text and the step we decide to use.

Let us consider an example:

```python
text_example = 'in another moment down went Alice after'.lower()
seq_len = 30
step = 2
divided = []
for i in range(0, len(text_example) - seq_len, step):
    divided.append(text_example[i : i + seq_len + 1])
divided
```

    ['in another moment down went ali',
     ' another moment down went alice',
     'nother moment down went alice a',
     'ther moment down went alice aft',
     'er moment down went alice after']




If we now change the step size, we will obtain a different number of sequences:

```python
step = 3
divided = []
for i in range(0, len(text_example) - seq_len, step_example):
    divided.append(text_example[i : i + seq_len + 1])
divided
```

    ['in another moment down went ali',
     'another moment down went alice ',
     'ther moment down went alice aft']

So the smaller the step size, the more sequences we will obtain. Because of memory and time constrains, we are going to use a step size of 3.

## Preprocessing the text

For each of the posting, we want to prepend the 'start_char' and append the `end_char` tokens to it. Then, for each of the characters in the text, we want to replace it with its index. Remember that all the `bad_chars` will default to the same index.

```python
seq_len = 100
step  = 3

def process(doc):
    doc = start_char + doc.lower() + end_char
    return [
        [to_int_func(z) for z in doc[i:i + seq_len + 1]]
        for i in range(0, len(doc) - seq_len, step)
    ]

def to_int_func(char):
    # checking if it's a good or bad char
    if char not in char_to_int:
        char = unknown_char
    return char_to_int[char]
```

Also note that `process` returns a list of list, where each piece of text has length `seq_len + 1`, one more than expected. This is due to the fact, that we need positive example to train the network (we are doing supervised learning after all). Therefore, we are going to pick all but the last element as the input, and the last one as the desired output. In that way we can just slice a `numpy.array` which is always a very elegant way of doing it.

At this point we need to concatenate the output of the `process` function when mapped to the list of postings. A `flatmap` function is what we need. Unfortunately it's not in the standard library so, we are going to borrow [this](http://stackoverflow.com/a/20037408) implementation from StackOverflow


```python
import itertools
def flatmap(func, *iterable):
    return itertools.chain.from_iterable(map(func, *iterable))
```

We can finally process the text:


```python
import numpy as np

transofmed = np.array(list(flatmap(process, all_listings)))
transofmed.shape
```

    (5926775, 101)


Each line contains a sequence of length 101. Remember that the first 100 elements represent the feature vector, while the last one is the desired output. Since we have used `np.array` is now very easy to separate them.


```python
x_ = transofmed[:, :seq_len]
y_ = transofmed[:, seq_len]
```

Again, each line of `x_` contains the indexes of the characters contained in the sequence, and `y_` the corresponding output.

Since we need a 3 dimensional tensor, we need to transform that via a one hot encoding:


```python
X_ = np.zeros((len(x_), seq_len, len(chars)), dtype=np.bool)
Y_ = np.zeros((len(y_), len(chars)), dtype=np.bool)


for time, sentence in enumerate(x_):
    for index, char_index in enumerate(sentence):
        X_[time, index, char_index] = 1
    Y_[time, y_[time]] = 1

```


```python
X_.shape, Y_.shape
```

# NOW THE MODEL

The last step is to actually create the model


```python
import random
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

```

    Using Theano backend.



```python
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(seq_len, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


```


```python
epochs = 100
model.fit(X_, Y_, batch_size=128, nb_epoch=epochs)
```

    Epoch 1/100
    453/453 [==============================] - 25s - loss: 3.4643
    Epoch 2/100
    453/453 [==============================] - 25s - loss: 3.3331
    Epoch 3/100
    453/453 [==============================] - 26s - loss: 3.2742
    Epoch 4/100
    453/453 [==============================] - 27s - loss: 3.2573
    Epoch 5/100
    453/453 [==============================] - 27s - loss: 3.3014
    Epoch 6/100
    453/453 [==============================] - 23s - loss: 3.2503
    Epoch 7/100
    453/453 [==============================] - 24s - loss: 3.2420
    Epoch 8/100
    453/453 [==============================] - 23s - loss: 3.2406
    Epoch 9/100
    453/453 [==============================] - 24s - loss: 3.2213
    Epoch 10/100
    453/453 [==============================] - 24s - loss: 3.2332
    Epoch 11/100
    453/453 [==============================] - 24s - loss: 3.2293
    Epoch 12/100
    453/453 [==============================] - 25s - loss: 3.2183
    Epoch 13/100
    453/453 [==============================] - 25s - loss: 3.2498
    Epoch 14/100
    453/453 [==============================] - 21s - loss: 3.1910
    Epoch 15/100
    453/453 [==============================] - 23s - loss: 3.1734
    Epoch 16/100
    453/453 [==============================] - 22s - loss: 3.1738
    Epoch 17/100
    453/453 [==============================] - 22s - loss: 3.1985
    Epoch 18/100
    453/453 [==============================] - 22s - loss: 3.1954
    Epoch 19/100
    453/453 [==============================] - 21s - loss: 3.1557
    Epoch 20/100
    453/453 [==============================] - 21s - loss: 3.1992
    Epoch 21/100
    453/453 [==============================] - 23s - loss: 3.2225
    Epoch 22/100
    453/453 [==============================] - 25s - loss: 3.2146
    Epoch 23/100
    453/453 [==============================] - 26s - loss: 3.1896
    Epoch 24/100
    453/453 [==============================] - 26s - loss: 3.1903
    Epoch 25/100
    453/453 [==============================] - 25s - loss: 3.1686
    Epoch 26/100
    453/453 [==============================] - 23s - loss: 3.1562
    Epoch 27/100
    453/453 [==============================] - 24s - loss: 3.1559
    Epoch 28/100
    453/453 [==============================] - 22s - loss: 3.1735
    Epoch 29/100
    453/453 [==============================] - 21s - loss: 3.1605
    Epoch 30/100
    453/453 [==============================] - 23s - loss: 3.1624
    Epoch 31/100
    453/453 [==============================] - 24s - loss: 3.1364
    Epoch 32/100
    453/453 [==============================] - 23s - loss: 3.5989
    Epoch 33/100
    453/453 [==============================] - 26s - loss: 3.1622
    Epoch 34/100
    453/453 [==============================] - 26s - loss: 3.1693
    Epoch 35/100
    453/453 [==============================] - 25s - loss: 3.1258
    Epoch 36/100
    453/453 [==============================] - 25s - loss: 3.1443
    Epoch 37/100
    453/453 [==============================] - 25s - loss: 3.1363
    Epoch 38/100
    453/453 [==============================] - 24s - loss: 3.1294
    Epoch 39/100
    453/453 [==============================] - 23s - loss: 3.1258
    Epoch 40/100
    453/453 [==============================] - 20s - loss: 3.0832
    Epoch 41/100
    453/453 [==============================] - 20s - loss: 3.1082
    Epoch 42/100
    453/453 [==============================] - 21s - loss: 3.1278
    Epoch 43/100
    453/453 [==============================] - 21s - loss: 3.1454
    Epoch 44/100
    453/453 [==============================] - 21s - loss: 3.1268
    Epoch 45/100
    453/453 [==============================] - 21s - loss: 3.1116
    Epoch 46/100
    453/453 [==============================] - 21s - loss: 3.1107
    Epoch 47/100
    453/453 [==============================] - 21s - loss: 3.0824
    Epoch 48/100
    453/453 [==============================] - 20s - loss: 3.0803
    Epoch 49/100
    453/453 [==============================] - 20s - loss: 3.0930
    Epoch 50/100
    453/453 [==============================] - 19s - loss: 3.0822
    Epoch 51/100
    453/453 [==============================] - 20s - loss: 3.0846
    Epoch 52/100
    453/453 [==============================] - 19s - loss: 3.0607
    Epoch 53/100
    453/453 [==============================] - 19s - loss: 3.0125
    Epoch 54/100
    453/453 [==============================] - 19s - loss: 3.0341
    Epoch 55/100
    453/453 [==============================] - 19s - loss: 3.0827
    Epoch 56/100
    453/453 [==============================] - 19s - loss: 3.0369
    Epoch 57/100
    453/453 [==============================] - 20s - loss: 2.9980
    Epoch 58/100
    453/453 [==============================] - 19s - loss: 2.9776
    Epoch 59/100
    453/453 [==============================] - 22s - loss: 2.9329
    Epoch 60/100
    453/453 [==============================] - 22s - loss: 2.9052
    Epoch 61/100
    453/453 [==============================] - 23s - loss: 2.9625
    Epoch 62/100
    453/453 [==============================] - 22s - loss: 2.9035
    Epoch 63/100
    453/453 [==============================] - 22s - loss: 2.8540
    Epoch 64/100
    453/453 [==============================] - 22s - loss: 2.8026
    Epoch 65/100
    453/453 [==============================] - 21s - loss: 2.8213
    Epoch 66/100
    453/453 [==============================] - 21s - loss: 2.7341
    Epoch 67/100
    453/453 [==============================] - 21s - loss: 2.6614
    Epoch 68/100
    453/453 [==============================] - 21s - loss: 2.8115
    Epoch 69/100
    453/453 [==============================] - 21s - loss: 2.6927
    Epoch 70/100
    453/453 [==============================] - 21s - loss: 2.5681
    Epoch 71/100
    453/453 [==============================] - 23s - loss: 2.6053
    Epoch 72/100
    453/453 [==============================] - 22s - loss: 2.5115
    Epoch 73/100
    453/453 [==============================] - 22s - loss: 2.4335
    Epoch 74/100
    453/453 [==============================] - 22s - loss: 2.3403
    Epoch 75/100
    453/453 [==============================] - 21s - loss: 2.4015
    Epoch 76/100
    453/453 [==============================] - 21s - loss: 2.2215
    Epoch 77/100
    453/453 [==============================] - 22s - loss: 2.2463
    Epoch 78/100
    453/453 [==============================] - 21s - loss: 2.1464
    Epoch 79/100
    453/453 [==============================] - 21s - loss: 2.1830
    Epoch 80/100
    453/453 [==============================] - 21s - loss: 2.0234
    Epoch 81/100
    453/453 [==============================] - 21s - loss: 1.9162
    Epoch 82/100
    453/453 [==============================] - 21s - loss: 1.9917
    Epoch 83/100
    453/453 [==============================] - 21s - loss: 1.8317
    Epoch 84/100
    453/453 [==============================] - 21s - loss: 1.6589
    Epoch 85/100
    453/453 [==============================] - 23s - loss: 1.7252
    Epoch 86/100
    453/453 [==============================] - 22s - loss: 1.6115
    Epoch 87/100
    453/453 [==============================] - 22s - loss: 1.5207
    Epoch 88/100
    453/453 [==============================] - 22s - loss: 1.5734
    Epoch 89/100
    453/453 [==============================] - 22s - loss: 1.5046
    Epoch 90/100
    453/453 [==============================] - 21s - loss: 1.2658
    Epoch 91/100
    453/453 [==============================] - 22s - loss: 1.2499
    Epoch 92/100
    453/453 [==============================] - 22s - loss: 1.3204
    Epoch 93/100
    453/453 [==============================] - 22s - loss: 1.3984
    Epoch 94/100
    453/453 [==============================] - 21s - loss: 1.0829
    Epoch 95/100
    453/453 [==============================] - 21s - loss: 1.1477
    Epoch 96/100
    453/453 [==============================] - 20s - loss: 1.0049
    Epoch 97/100
    453/453 [==============================] - 21s - loss: 0.9370
    Epoch 98/100
    453/453 [==============================] - 22s - loss: 1.0782
    Epoch 99/100
    453/453 [==============================] - 23s - loss: 0.7462
    Epoch 100/100
    453/453 [==============================] - 22s - loss: 0.8366





    <keras.callbacks.History at 0x121863f60>




```python

```
