# definir arquitecturas de MLP, LSTM y CNN temporal para clasificacion

from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Input
from keras.optimizers import Adam

class MLPModel:
    def __init__(self, input_shape, num_classes):
        self.model = Sequential()
        self.model.add(Input(shape=input_shape))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    def fit(self, X, y, epochs=10, batch_size=32, **kwargs):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, **kwargs)
    
    def predict(self, X):
        return self.model.predict(X)

class LSTMModel: # revisar arquitectura
    def __init__(self, input_shape, num_classes):
        self.model = Sequential()
        self.model.add(Input(shape=input_shape))
        self.model.add(LSTM(128, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    def fit(self, X, y, epochs=10, batch_size=32, **kwargs):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, **kwargs)
    
    def predict(self, X):
        return self.model.predict(X)

class CNNModel:
    def __init__(self, input_shape, num_classes):
        self.model = Sequential()
        self.model.add(Input(shape=input_shape))
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    def fit(self, X, y, epochs=10, batch_size=32, **kwargs):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, **kwargs)
    
    def predict(self, X):
        return self.model.predict(X)