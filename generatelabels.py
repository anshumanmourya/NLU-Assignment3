
# coding: utf-8

# In[3]:


import numpy as np
import nltk
import pycrfsuite
from bs4 import BeautifulSoup as bs
from bs4.element import Tag


# In[4]:


import sys
program_name = sys.argv[0]
infilename = sys.argv[1]
outfilename = sys.argv[2]
docs1 = []
docs2 = []
docs = []
with open(infilename) as fp:
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

data = []
for i, doc in enumerate(docs):

    tokens = [t for t in doc]
    tagged = nltk.pos_tag(tokens)
    data.append([(w, pos) for w, (word, pos) in zip(doc, tagged)])

#print data


# In[ ]:


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




# In[5]:


X_test = [extract_features(doc) for doc in data]
tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]

labels = {"T": 2, "O": 1,"D" : 0}

predictions = np.array([labels[tag] for row in y_pred for tag in row])
np.savetxt('outfilename',predictions)

