from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# definir clases para random forest, KNN, SVM para clasificación y ser llamado en models/train_classic.py
class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators,
                "random_state": self.random_state}

class KNNModel:
    def __init__(self, n_neighbors=6):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_params(self, deep=True):
        return {'n_neighbors': self.n_neighbors}

class SVMModel:
    def __init__(self, kernel='linear', C=1.0, probability=True):
        self.kernel = kernel
        self.C = C
        self.probability = probability
        self.model = SVC(kernel=kernel, C=C, probability=probability, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if not self.probability:
            raise RuntimeError("SVM must be initialized with probability=True to use predict_proba")
        return self.model.predict_proba(X)
    
    def get_params(self, deep=True):
        return {"kernel": self.kernel, "C": self.C, "probability": self.probability}
