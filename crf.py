
# coding: utf-8

# In[40]:


import codecs
import numpy as np
import nltk
import pycrfsuite
from bs4 import BeautifulSoup as bs
from bs4.element import Tag
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[41]:


docs1 = []
docs2 = []
docs = []
with open('/home/anshuman/Downloads/ner.txt') as fp:
    docs1 = fp.read().splitlines()
for item in docs1:
    t = tuple(item.split())
    docs2.append(t)
doc = []
for t in docs2:
    if t != ():
        doc.append(t)
    else:
        docs.append(doc)
        doc = []


print(docs)

data = []
for i, doc in enumerate(docs):

    tokens = [t for t, label in doc]
    tagged = nltk.pos_tag(tokens)
    data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])

#print data


# In[42]:


def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

    # Common features for all words
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'word.isalpha=%s' % word.isalpha(),
        'postag=' + postag
    ]
    if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1word.isalpha=%s' % word.isalpha(),
            '-1:postag=' + postag1
        ])
    else:
        features.append('BOS')

    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1word.isalpha=%s' % word.isalpha(),
            '+1:postag=' + postag1
        ])
    else:
        features.append('EOS')

    return features

def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]
def get_labels(doc):
    return [label for (token, postag, label) in doc]


X = [extract_features(doc) for doc in data]
y = [get_labels(doc) for doc in data]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 7)

trainer = pycrfsuite.Trainer(verbose=True)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,

    # coefficient for L2 penalty
    'c2': 0.01,  

    # maximum number of iterations
    'max_iterations': 200,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
})

trainer.train('crf.model')
tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]

i = 12
for yt,x, y in zip(y_test[i] ,y_pred[i], [x[1].split("=")[1] for x in X_test[i]]):
    print("%s (%s) %s" % (y, x, yt))

labels = {"T": 2, "O": 1,"D" : 0}

predictions = np.array([labels[tag] for row in y_pred for tag in row])
truths = np.array([labels[tag] for row in y_test for tag in row])
accuracy = accuracy_score(truths, predictions)
print(accuracy)



print(classification_report(
    truths, predictions,
target_names=["T", "O","D"]))



# In[45]:


y_pred = [tagger.tag(xseq) for xseq in X_test]

# Let's take a look at a random sample in the testing set
i = 1
for yt,x, y in zip(y_test[i] ,y_pred[i], [x[1].split("=")[1] for x in X_test[i]]):
    print("%s (%s) %s" % (y, x, yt))

# Create a mapping of labels to indices
labels = {"T": 2, "O": 1,"D" : 0}

