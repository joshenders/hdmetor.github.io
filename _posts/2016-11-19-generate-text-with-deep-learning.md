---
layout: post
title: Generate a new job posting à la Hacker News with LSTM and Keras
description: "Use deeplearning library keras to generate a new job posting HacerkNews style"
modified: 2016-11-19
tags: [python, keras, deep learning, hacker news]
image:
  feature: abstract-4.jpg
  credit: dargadgetz
  creditlink: http://www.dargadgetz.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/
---
In this blog, we are going to show how to train a charachter-level deep learning model to hallucinate new jobs posting.

In order to train the model, we need data from the previous Hacker news posting. From now on, we are assuming that such data in contained in a local Elasticsearch index, as descried in [this]({% post_url 2016-10-07-how-i-found-my-job-in-sf-using-hackernews-and-elasticsearch %}) blog post.
To gather the data, we only need to perform a 'match all' query and grab the text field from the source.


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

Note that we used the function `scan` (found in the `helpers` module) rather than `search` because of the potentially large output.

## Organizing the text and counting the characters

Since we want to train a character level network, we need create a mapping from character to integers. Training the network on _all_ the character will be way too wasteful (if possible at all), and produce very noisy results. Therefore we are going to perform the following steps:

- use only one type of apostrophe

- lowercase all the text

- define a subset off all the characters (`bad_chars`) which will contain the least used characters (note that the threshold for that was selected manually after printing `counter.most_common()`)

- define `good_chars` (i.e. the characters that will actually go into the model) as the complementary set to `bad_chars`

- pick a special `start_char`, `end_char` and `unknown_char` after making sure they don't belong in the `good_char` set

- replace each character form the `bad_char` listing with the `unknown_char` one

The reason why we need the start and stop tokens is that we want to be able to generate a full listing (i.e. from beginning to end), so we need to teach the model when a listing starts and ends.

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

good_chars.extend([start_char, end_char, unknown_char])
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
for i in range(0, len(text_example) - seq_len, step):
    divided.append(text_example[i : i + seq_len + 1])
divided
```

    ['in another moment down went ali',
     'another moment down went alice ',
     'ther moment down went alice aft']

So the smaller the step size, the more sequences we will obtain. Because of memory and time constrains, we are going to use a step size of 3.

## Preprocessing the text

For each of the posting, we want to perpend the `start_char` and append the `end_char` tokens to it. Then, for each of the characters in the text, we want to replace it with its index. Remember that each char in the `bad_chars` set will default to the same index.

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

Also note that `process` returns a list of list, where each piece of text has length `seq_len + 1`, one more than expected. This is due to the fact, that we need positive example to train the network (we are doing supervised learning after all). Therefore, we are going to pick all but the last element as the input, and the last one as the desired output. To obtains this, we can just slice a `numpy.array` which is always a very elegant and concise way.

At this point we need to concatenate the output of the `process` function when mapped to the list of postings. A `flatmap` function is what we need. Unfortunately it's not in the standard library so, we are going to borrow [this](http://stackoverflow.com/a/20037408) implementation from StackOverflow
:


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

To match the input dimensions, we are going to use a one-hot encoding for our data.

```python
X = np.zeros((len(x_), seq_len, len(good_chars)), dtype=np.bool)
Y = np.zeros((len(y_), len(good_chars)), dtype=np.bool)

for time, sentence in enumerate(x_):
    for index, char_index in enumerate(sentence):
        X[time, index, char_index] = 1
    Y[time, y_[time]] = 1
X.shape, Y.shape
```
    ((5926775, 100, 63), (5926775, 63))

# Defining the Deep Learning model

Now that our data is ready, we can just define the model in Keras and start training:

```python
import random
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(seq_len, len(good_chars))))
model.add(Dropout(0.4))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.4))
model.add(Dense(len(good_chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```

This is the beauty of Keras, we used only 8 lines to build our model in a Lego like fashion.

The model we used here is 2 layers of LSTM with dropout (with probability 0.4 each). The last layer is a dense layer of dimension `len(good_char)`, one for each character (if you remember we used a the same character for a bunch of noisy elements). Note that we didn't use any embedding layer, because we did the vectorizing manually.

Let's start training and save and save the model each epoch:


```python

epochs = 100
batch_size = 128
for epoch in epochs:
  model.fit(X, Y, batch_size=batch_size, nb_epoch=1)
  file_name = '{}.hdf5'.format(epoch)
  model.save(file_name)
```

Time to wait for the model to be trained. Please come back later for a new blog with some (hopefully interesting) examples of hallucinated job posting.
