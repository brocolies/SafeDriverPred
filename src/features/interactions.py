from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

class InteractionFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # fit 단계에서는 아무것도 학습할 필요가 없으므로 self를 그대로 반환
        return self

    def transform(self, X):
        # 입력받은 데이터프레임 X를 복사해서 원본을 보호
        X_new = X.copy()
        
        # 피처 엔지니어링

        return X_new
