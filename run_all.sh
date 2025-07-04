#!/bin/bash
python3 models/train_rf.py
python3 models/train_knn.py
python3 models/train_svm.py
python3 models/train_mlp.py
python3 models/train_lstm.py
python3 models/train_cnn.py