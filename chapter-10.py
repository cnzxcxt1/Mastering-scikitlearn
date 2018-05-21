# chapter 10: From the Perceptron to Artificial Neural Networks
# Approximating XOR with Multilayer perceptrons

from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
y = [0, 1, 1, 0] * 1000
X = [[0, 0], [0, 1], [1, 0], [1, 1]] * 1000
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

clf = MLPClassifier(hidden_layer_sizes = [2], activation='logistic', solver='adam', random_state=3)
clf.fit(X_train, y_train)

print('Number of layers: %s. Number of outputs: %s' % (clf.n_layers_, clf.n_outputs_))
predictions = clf.predict(X_test)
print('Accuracy:', clf.score(X_test, y_test))
for i, p in enumerate(predictions[:10]):
    print('True: %s, Predicted: %s' % (y_test[i], p))


# Classifying handwritten digits
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

if __name__ == '__main__':
    digits = load_digits()
    X = digits.data
    y = digits.target
    pipeline = Pipeline([
        ('ss', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=[150, 100], alpha=0.1))
    ])
    print(cross_val_score(pipeline, X, y, n_jobs=-1))

