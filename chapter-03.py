### chapter 03: Feature Extraction and Preprocessing
### Extracting features from categorical variables
import numpy as np
from sklearn.feature_extraction import DictVectorizer
onehot_encoder = DictVectorizer()
instances = [
    {'city': 'New York'},
    {'city': 'San Francisco'},
    {'city': 'Chapel Hill'}
]

print(onehot_encoder.fit_transform(instances).toarray())


### Extracting features from text
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game'
]

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game'
]
vectorizer = CountVectorizer()
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)


corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

from sklearn.metrics.pairwise import euclidean_distances
#counts = [
#    [0, 1, 1, 0, 0, 1, 0, 1],
#    [0, 1, 1, 1, 1, 0, 0, 0],
#    [1, 0, 0, 0, 0, 0, 1, 0]
#]

vectorizer = CountVectorizer()
# in this way, the variable counts should be a matrix
counts = vectorizer.fit_transform(corpus).todense()


print('Distance between 1st and 2nd documents:', euclidean_distances(counts[0], counts[1]))
print('Distance between 1st and 3rd documents:', euclidean_distances(counts[0], counts[2]))
print('Distance between 2nd and 3rd documents:', euclidean_distances(counts[1], counts[2]))

################# Sample 1 #################
from sklearn.feature_extraction import DictVectorizer
onehot_encoder = DictVectorizer()
instances = [
    {'city': 'New York'},
    {'city': 'San Francisco'},
    {'city': 'Chapel Hill'}
]
print(onehot_encoder.fit_transform(instances).toarray())


################# Sample 2 #################

corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game'
]

################# Sample 3 #################

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game'
]
vectorizer = CountVectorizer(binary=True)
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)


################# Sample 4 #################

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(corpus).todense()
print(X)
print(vectorizer.vocabulary_)
for i, document in enumerate(corpus):
    print(document, '=', X[i])


################# Sample 6: Stop-word filtering #################

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]
vectorizer = CountVectorizer(binary=True, stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)


################# Sample 7 #################


from sklearn.metrics.pairwise import euclidean_distances
counts = [
    [0, 1, 1, 0, 0, 1, 0, 1],
    [0, 1, 1, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0]
]
counts = np.matrix(counts)
print('Distance between 1st and 2nd documents:', euclidean_distances(counts[0], counts[1]))
print('Distance between 1st and 3rd documents:', euclidean_distances(counts[0], counts[2]))
print('Distance between 2nd and 3rd documents:', euclidean_distances(counts[1], counts[2]))

################# Sample 8: Stemming and lemmatization #################

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'He ate the sandwiches',
    'Every sandwich was eaten by him'
]
vectorizer = CountVectorizer(binary = True, stop_words = 'english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

################# Sample 7 #################
corpus = [
    'I am gathering ingredients for the sandwich.',
    'There were many wizards at the gathering.'
]

#import nltk
#nltk.download('wordnet')
################# Sample 8 #################
# need to install NLTK package
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('gathering', 'v'))
print(lemmatizer.lemmatize('gathering', 'n'))


################# Sample 8 #################

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
print(stemmer.stem('gathering'))


################# Sample 9 #################

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
wordnet_tags = ['n', 'v']
corpus = [
    'He ate the sandwiches',
    'Every sandwich was eaten by him'
]

#nltk.download('punkt')
stemmer = PorterStemmer()
print('Stemmed:', [[stemmer.stem(token) for token in word_tokenize(document)] for document in corpus])


def lemmatize(token, tag):
    if tag[0].lower() in ['n', 'v']:
        return lemmatizer.lemmatize(token, tag[0].lower())
    return token
# nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()
tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]
print('Lemmatized:', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])


################# Sample 10: Extending bag-of-words with TF-IDF weights #################

from sklearn.feature_extraction.text import CountVectorizer
corpus = ['The dog ate a sandwich, the wizard transfigured a sandwich, and I ate a sandwich']
vectorizer = CountVectorizer(stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)


################# Sample 10 #################

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
corpus = [
    'The dog foo bar dog dog dog dog foo bar',
    'Dog the hat'
]

vectorizer_01 = CountVectorizer(stop_words='english')
vectorizer_02 = TfidfVectorizer(stop_words='english')

print(vectorizer_01.fit_transform(corpus).todense())


################# Sample 11 #################

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
corpus = [
    'The dog ate a sandwich and I ate a sandwich',
    'The wizard transfigured a sandwich'
]
vectorizer = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(use_idf=False)
X = vectorizer.fit_transform(corpus)
print('Count vectors:\n', X.todense())
print('Vocabulary:\n', vectorizer.vocabulary_)
print('TF vectors:\n', transformer.fit_transform(X).todense())


################# Sample 12 #################

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'The dog ate a sandwich and I ate a sandwich',
    'The wizard transfigured a sandwich'
]
vectorizer = TfidfVectorizer(stop_words='english')
print(vectorizer.fit_transform(corpus))


################# Sample 13: Space-efficient feature vectorizing with the hashing trick #################

from sklearn.feature_extraction.text import HashingVectorizer
corpus = ['the', 'ate', 'bacon', 'cat']
vectorizer = HashingVectorizer(n_features=6)
print(vectorizer.transform(corpus).todense())


### Extracting features from images
################# Figure 14 #################

from sklearn import datasets
import matplotlib.pyplot as plt
digits = datasets.load_digits()
print('Digit:', digits.target[0])
print(digits.images[0])
plt.figure()
plt.axis('off')
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

### Extracting points of interest as features
################# Sample 14 #################

from sklearn import datasets
digits = datasets.load_digits()
print('Digit:', digits.target[0])
print(digits.images[0])
print('Feature vector:\n', digits.images[0].reshape(-1, 64))

################# Sample 15 #################


import numpy as np
from skimage.feature import corner_harris, corner_peaks
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.exposure import equalize_hist


def show_corners(corners, image):
    fig = plt.figure()
    plt.gray()
    plt.imshow(image)
    y_corner, x_corner = zip(*corners)
    plt.plot(x_corner, y_corner, 'or')
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5)
    plt.show()

mandrill = io.imread('mandrill.png')
mandrill = equalize_hist(rgb2gray(mandrill))
corners = corner_peaks(corner_harris(mandrill), min_distance=2)
show_corners(corners, mandrill)


################# Sample 16: SIFT and SURF #################
### this code has some problem
import mahotas as mh
from mahotas.features import surf

image = mh.imread('zipper.jpg', as_grey=True)
print('The first SURF descriptor:\n', surf.surf(image)[0])
print('Extracted %s SURF descriptors' % len(surf.surf(image)))


################# Sample 17 #################

from sklearn import preprocessing
import numpy as np
X = np.array([
    [0., 0., 5., 13., 9., 1.],
    [0., 0., 13., 15., 10., 15.],
    [0., 3., 15., 2., 0., 11.]
])
print(preprocessing.scale(X))
