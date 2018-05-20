################# Figures 05_01 #################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report

instances = [
    {'plays fetch': True, 'species': 'Dog'},
    {'plays fetch': False, 'species': 'Dog'},
    {'plays fetch': False, 'species': 'Cat'},
    {'plays fetch': False, 'species': 'Cat'},
    {'plays fetch': False, 'species': 'Cat'},
    {'plays fetch': False, 'species': 'Cat'},
    {'plays fetch': False, 'species': 'Cat'},
    {'plays fetch': False, 'species': 'Dog'},
    {'plays fetch': False, 'species': 'Cat'},
    {'plays fetch': True, 'species': 'Dog'},
    {'plays fetch': True, 'species': 'Dog'},
    {'plays fetch': False, 'species': 'Cat'},
    {'plays fetch': True, 'species': 'Cat'},
    {'plays fetch': True, 'species': 'Dog'}
]
df = pd.DataFrame(instances)

X_train = [[1] if a else [0] for a in df['plays fetch']]
y_train = [1 if d == 'Dog' else 0 for d in df['species']]
labels = ['plays fetch']

clf = DecisionTreeClassifier(max_depth=None, max_features=None, criterion='entropy', min_samples_leaf=1, min_samples_split=2)

print(X_train)
clf.fit(X_train, y_train)

f = 'tree.dot'
export_graphviz(clf, out_file=f, feature_names=labels)
# 应该是没有权限才会这样的


################# Figures 05_02 #################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report

instances = [
    {'is grumpy': False, 'species': 'Dog'},
    {'is grumpy': True, 'species': 'Dog'},
    {'is grumpy': True, 'species': 'Cat'},
    {'is grumpy': True, 'species': 'Cat'},
    {'is grumpy': False, 'species': 'Cat'},
    {'is grumpy': True, 'species': 'Cat'},
    {'is grumpy': True, 'species': 'Cat'},
    {'is grumpy': False, 'species': 'Dog'},
    {'is grumpy': True, 'species': 'Cat'},
    {'is grumpy': False, 'species': 'Dog'},
    {'is grumpy': False, 'species': 'Dog'},
    {'is grumpy': False, 'species': 'Cat'},
    {'is grumpy': True, 'species': 'Cat'},
    {'is grumpy': True, 'species': 'Dog'}
]
df = pd.DataFrame(instances)
X_train = [[1] if a else [0] for a in df['is grumpy']]
y_train = [1 if d == 'Dog' else 0 for d in df['species']]
labels = ['is grumpy']
clf = DecisionTreeClassifier(max_depth=None, max_features=None, criterion='entropy', min_samples_leaf=1, min_samples_split=2)
clf.fit(X_train, y_train)

f = 'is-grumpy.dot'
export_graphviz(clf, out_file=f, feature_names=labels)


################# Figures 05_03 #################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report

instances = [
    {'favorite food': False, 'species': 'Dog'},
    {'favorite food': False, 'species': 'Dog'},
    {'favorite food': True, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': True, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': True, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Dog'},
    {'favorite food': True, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Dog'},
    {'favorite food': False, 'species': 'Dog'},
    {'favorite food': True, 'species': 'Cat'},
    {'favorite food': True, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Dog'}
]
df = pd.DataFrame(instances)
X_train = df[['favorite food']]

vectorizer = DictVectorizer()
X_train = [[1] if a else [0] for a in df['favorite food']]
y_train = [1 if d == 'Dog' else 0 for d in df['species']]

labels = ['favorite food=cat food']

clf = DecisionTreeClassifier(max_depth=None, max_features=None, criterion='entropy', min_samples_leaf=1, min_samples_split=2)
print(X_train)
clf.fit(X_train, y_train)

f = 'favorite-food-cat-food.dot'
export_graphviz(clf, out_file=f, feature_names=labels)


################# Figures 05_04 #################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report

instances = [
    {'favorite food': False, 'species': 'Dog'},
    {'favorite food': True, 'species': 'Dog'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': True, 'species': 'Dog'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': True, 'species': 'Dog'},
    {'favorite food': False, 'species': 'Dog'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Dog'}
]
df = pd.DataFrame(instances)
X_train = [[1] if a else [0] for a in df['favorite food']]
y_train = [1 if d == 'Dog' else 0 for d in df['species']]
labels = ['favorite food=dog food']
clf = DecisionTreeClassifier(max_depth=None, max_features=None, criterion='entropy', min_samples_leaf=1, min_samples_split=2)
clf.fit(X_train, y_train)

f = 'favorite-food-dog-food.dot'
export_graphviz(clf, out_file=f, feature_names=labels)


################# Figures 05_05 #################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report

instances = [
    {'favorite food': True, 'species': 'Dog'},
    {'favorite food': False, 'species': 'Dog'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': True, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': True, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Dog'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Dog'},
    {'favorite food': True, 'species': 'Dog'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': True, 'species': 'Dog'}
]
df = pd.DataFrame(instances)
X_train = [[1] if a else [0] for a in df['favorite food']]
y_train = [1 if d == 'Dog' else 0 for d in df['species']]
labels = ['favorite food=bacon']
clf = DecisionTreeClassifier(max_depth=None, max_features=None, criterion='entropy',  min_samples_leaf=1, min_samples_split=2)
clf.fit(X_train, y_train)

f = 'favorite-food-bacon.dot'
export_graphviz(clf, out_file=f, feature_names=labels)


################# Data #################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer

instances = [
    {'plays fetch': True, 'species': 'Dog'},
    {'plays fetch': False, 'species': 'Dog'},
    {'plays fetch': False, 'species': 'Cat'},
    {'plays fetch': False, 'species': 'Cat'},
    {'plays fetch': False, 'species': 'Dog'},
    {'plays fetch': True, 'species': 'Dog'},
    {'plays fetch': True, 'species': 'Dog'},
    {'plays fetch': True, 'species': 'Dog'}
]
df = pd.DataFrame(instances)
X_train = [[1] if a else [0] for a in df['plays fetch']]
y_train = [1 if d == 'Dog' else 0 for d in df['species']]
labels = ['plays fetch']
clf = DecisionTreeClassifier(max_depth=None, max_features=None, criterion='entropy', min_samples_leaf=1, min_samples_split=2)
clf.fit(X_train, y_train)
f = 'level2-play-fetch.dot'
export_graphviz(clf, out_file=f, feature_names=labels)


################# Data #################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer

instances = [
    {'is grumpy': False, 'species': 'Dog'},
    {'is grumpy': True, 'species': 'Dog'},
    {'is grumpy': True, 'species': 'Cat'},
    {'is grumpy': True, 'species': 'Cat'},
    {'is grumpy': False, 'species': 'Dog'},
    {'is grumpy': False, 'species': 'Dog'},
    {'is grumpy': False, 'species': 'Dog'},
    {'is grumpy': True, 'species': 'Dog'}
]
df = pd.DataFrame(instances)
X_train = [[1] if a else [0] for a in df['is grumpy']]
y_train = [1 if d == 'Dog' else 0 for d in df['species']]
labels = ['is grumpy']
clf = DecisionTreeClassifier(max_depth=None, max_features=None, criterion='entropy', min_samples_leaf=1, min_samples_split=2)
clf.fit(X_train, y_train)
f = 'level2-is-grumpy.dot'
export_graphviz(clf, out_file=f, feature_names=labels)

################# Data #################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer

instances = [
    {'favorite food': False, 'species': 'Dog'},
    {'favorite food': True, 'species': 'Dog'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': True, 'species': 'Dog'},
    {'favorite food': True, 'species': 'Dog'},
    {'favorite food': False, 'species': 'Dog'},
    {'favorite food': False, 'species': 'Dog'}
]
df = pd.DataFrame(instances)
X_train = [[1] if a else [0] for a in df['favorite food']]
y_train = [1 if d == 'Dog' else 0 for d in df['species']]
labels = ['favorite food=dog food']
clf = DecisionTreeClassifier(max_depth=None, max_features=None, criterion='entropy', min_samples_leaf=1, min_samples_split=2)
clf.fit(X_train, y_train)
f = 'level2-favorite-food-dog-food.dot'
export_graphviz(clf, out_file=f, feature_names=labels)

################# Data #################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz

instances = [
    {'favorite food': True, 'species': 'Dog'},
    {'favorite food': False, 'species': 'Dog'},
    {'favorite food': True, 'species': 'Cat'},
    {'favorite food': True, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Dog'},
    {'favorite food': False, 'species': 'Dog'},
    {'favorite food': True, 'species': 'Dog'},
    {'favorite food': True, 'species': 'Dog'}
]
df = pd.DataFrame(instances)
X_train = [[1] if a else [0] for a in df['favorite food']]
y_train = [1 if d == 'Dog' else 0 for d in df['species']]
labels = ['favorite food=bacon']
clf = DecisionTreeClassifier(max_depth=None, max_features=None, criterion='entropy', min_samples_leaf=1, min_samples_split=2)
clf.fit(X_train, y_train)
f = 'level2-favorite-food-bacon.dot'
export_graphviz(clf, out_file=f, feature_names=labels)


################# Data #################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz

instances = [
    {'favorite food': False, 'species': 'Dog'},
    {'favorite food': True, 'species': 'Cat'},
    {'favorite food': True, 'species': 'Cat'},
    {'favorite food': True, 'species': 'Dog'}
]
df = pd.DataFrame(instances)
X_train = [[1] if a else [0] for a in df['favorite food']]
y_train = [1 if d == 'Dog' else 0 for d in df['species']]
labels = ['favorite food=bacon']
clf = DecisionTreeClassifier(max_depth=None, max_features=None, criterion='entropy', min_samples_leaf=1, min_samples_split=2)
clf.fit(X_train, y_train)
f = 'level3-favorite-food-bacon.dot'
export_graphviz(clf, out_file=f, feature_names=labels)


################# Data #################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz

instances = [
    {'favorite food': True, 'species': 'Dog'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Cat'},
    {'favorite food': False, 'species': 'Dog'}
]
df = pd.DataFrame(instances)
X_train = [[1] if a else [0] for a in df['favorite food']]
y_train = [1 if d == 'Dog' else 0 for d in df['species']]
labels = ['favorite food=dog food']
clf = DecisionTreeClassifier(max_depth=None, max_features=None, criterion='entropy', min_samples_leaf=1, min_samples_split=2)
clf.fit(X_train, y_train)
f = 'level3-favorite-food-dog-food.dot'
export_graphviz(clf, out_file=f, feature_names=labels)

################# Data #################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz

instances = [
    {'plays fetch': False, 'species': 'Dog'},
    {'plays fetch': False, 'species': 'Cat'},
    {'plays fetch': False, 'species': 'Cat'},
    {'plays fetch': True, 'species': 'Dog'}
]
df = pd.DataFrame(instances)
X_train = [[1] if a else [0] for a in df['plays fetch']]
y_train = [1 if d == 'Dog' else 0 for d in df['species']]
labels = ['plays fetch']
clf = DecisionTreeClassifier(max_depth=None, max_features=None, criterion='entropy', min_samples_leaf=1, min_samples_split=2)
clf.fit(X_train, y_train)
f = 'level3-plays-fetch.dot'
export_graphviz(clf, out_file=f, feature_names=labels)


################# Data #################



################# Sample 1: Ad classification with Decision Trees #################

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

if __name__ == '__main__':
    df = pd.read_csv('ad.data', header=None)
    explanatory_variable_columns = set(df.columns.values)
    response_variable_column = df[len(df.columns.values)-1]
    # The last column describes the targets
    explanatory_variable_columns.remove(len(df.columns.values)-1)

    y = [1 if e == 'ad.' else 0 for e in response_variable_column]
    X = df[list(explanatory_variable_columns)]
    X.replace(to_replace=' *\?', value=-1, regex=True, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipeline = Pipeline([
        ('clf', DecisionTreeClassifier(criterion='entropy'))
    ])
    parameters = {
        'clf__max_depth': (150, 155, 160),
        'clf__min_samples_split': (2, 3, 4),
        'clf__min_samples_leaf': (1, 2, 3)
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='f1')
    grid_search.fit(X_train, y_train)
    print('Best score: %0.3f' % grid_search.best_score_)
    print('Best parameters set:')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))

    predictions = grid_search.predict(X_test)
    print(classification_report(y_test, predictions))
    print(grid_search.score(X_test, y_test))


################# Sample 2: Ad classification with Random Forests #################

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

if __name__ == '__main__':
    df = pd.read_csv('ad.data', header=None)
    explanatory_variable_columns = set(df.columns.values)
    response_variable_column = df[len(df.columns.values)-1]
    # The last column describes the targets
    explanatory_variable_columns.remove(len(df.columns.values)-1)

    y = [1 if e == 'ad.' else 0 for e in response_variable_column]
    X = df[list(explanatory_variable_columns)]
    X.replace(to_replace=' *\?', value=-1, regex=True, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipeline = Pipeline([
        ('clf', RandomForestClassifier(criterion='entropy'))
    ])
    parameters = {
        'clf__n_estimators': (5, 10, 20, 50),
        'clf__max_depth': (50, 150, 250),
        'clf__min_samples_split': (2, 3, 4),
        'clf__min_samples_leaf': (1, 2, 3)
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='f1')
    grid_search.fit(X_train, y_train)
    print('Best score: %0.3f' % grid_search.best_score_)
    print('Best parameters set:')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))

    predictions = grid_search.predict(X_test)
    print(classification_report(y_test, predictions))
