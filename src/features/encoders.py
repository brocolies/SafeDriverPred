from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Categorical features 전용 인코더
    - One-Hot Encoding 적용
    - 익명화된 categorical 피처들의 순서 관계 제거
    - 실무에서 검증된 방식 (Porto Seguro 1위 솔루션 기반)
    """
    
    def __init__(self, cat_cols=None, handle_unknown='ignore', drop='first'):
        """
        Parameters:
        -----------
        cat_cols : list, optional
            categorical로 처리할 컬럼들. None이면 자동 감지
        handle_unknown : str, default='ignore'
            test set에서 새로운 카테고리 처리 방식
        drop : str, default='first'
            multicollinearity 방지를 위한 첫 번째 카테고리 제거
        """
        self.cat_cols = cat_cols
        self.handle_unknown = handle_unknown
        self.drop = drop
        self.encoder = None
        self.feature_names_out = None
        
    def _identify_categorical_cols(self, X):
        """
        자동으로 categorical 컬럼 식별
        Porto Seguro 데이터 기준: 'cat'이 포함된 컬럼들
        """
        if self.cat_cols is not None:
            return self.cat_cols
        
        # 'cat'이 포함된 컬럼들 자동 감지
        cat_cols = [col for col in X.columns if 'cat' in col.lower()]
        
        if len(cat_cols) == 0:
            print("Warning: No categorical columns found. Check column names.")
        else:
            print(f"Auto-detected categorical columns: {cat_cols}")
            
        return cat_cols
    
    def fit(self, X, y=None):
        """
        Categorical columns에 대해 One-Hot Encoder 학습
        
        Parameters:
        -----------
        X : pd.DataFrame
            학습 데이터
        y : array-like, optional
            타겟 변수 (사용되지 않음)
            
        Returns:
        --------
        self : CategoricalEncoder
        """
        # Categorical 컬럼 식별
        self.cat_cols = self._identify_categorical_cols(X)
        
        if len(self.cat_cols) == 0:
            # Categorical 컬럼이 없으면 아무것도 하지 않음
            self.encoder = None
            return self
        
        # One-Hot Encoder 학습
        self.encoder = OneHotEncoder(
            handle_unknown=self.handle_unknown,
            drop=self.drop,
            sparse_output=False  # Dense array 반환
        )
        
        # Categorical 컬럼들만 선택해서 학습
        cat_data = X[self.cat_cols]
        self.encoder.fit(cat_data)
        
        # 변환 후 피처 이름 생성
        self._generate_feature_names()
        
        return self
    
    def _generate_feature_names(self):
        """
        One-Hot 인코딩 후 피처 이름 생성
        예: ps_ind_02_cat -> ps_ind_02_cat_1, ps_ind_02_cat_2, ...
        """
        if self.encoder is None:
            self.feature_names_out = []
            return
        
        feature_names = []
        
        for i, col in enumerate(self.cat_cols):
            # 각 컬럼의 카테고리별 이름 생성
            categories = self.encoder.categories_[i]
            
            # drop='first'인 경우 첫 번째 카테고리 제외
            if self.drop == 'first':
                categories = categories[1:]
            
            for cat in categories:
                feature_names.append(f"{col}_{cat}")
        
        self.feature_names_out = feature_names
    
    def transform(self, X):
        """
        Categorical columns을 One-Hot Encoding으로 변환
        
        Parameters:
        -----------
        X : pd.DataFrame
            변환할 데이터
            
        Returns:
        --------
        X_transformed : pd.DataFrame
            변환된 데이터 (원본 + One-Hot 인코딩된 피처들)
        """
        X_new = X.copy()
        
        # Categorical 컬럼이 없거나 encoder가 없으면 원본 반환
        if self.encoder is None or len(self.cat_cols) == 0:
            return X_new
        
        # Categorical 데이터 변환
        cat_data = X_new[self.cat_cols]
        encoded_data = self.encoder.transform(cat_data)
        
        # 변환된 데이터를 DataFrame으로 변환
        encoded_df = pd.DataFrame(
            encoded_data, 
            columns=self.feature_names_out,
            index=X_new.index
        )
        
        # 원본 categorical 컬럼들 제거
        X_new = X_new.drop(columns=self.cat_cols)
        
        # One-Hot 인코딩된 피처들 추가
        X_new = pd.concat([X_new, encoded_df], axis=1)
        
        return X_new
    
    def get_feature_names_out(self, input_features=None):
        """
        변환 후 피처 이름들 반환
        sklearn 호환성을 위한 메서드
        """
        if input_features is None:
            return self.feature_names_out
        
        # 원본 피처들에서 categorical 제거하고 새로운 피처들 추가
        remaining_features = [f for f in input_features if f not in self.cat_cols]
        return remaining_features + self.feature_names_out
    
    def get_categorical_info(self):
        """
        인코딩 정보 반환 (디버깅용)
        """
        if self.encoder is None:
            return "No categorical columns to encode"
        
        info = {}
        for i, col in enumerate(self.cat_cols):
            info[col] = {
                'categories': self.encoder.categories_[i].tolist(),
                'n_categories': len(self.encoder.categories_[i])
            }
        
        return info


# 사용 예시 및 테스트 함수
def test_categorical_encoder():
    """
    CategoricalEncoder 테스트 함수
    """
    # 샘플 데이터 생성
    data = pd.DataFrame({
        'ps_ind_02_cat': [1, 2, 3, 1, 2],
        'ps_ind_04_cat': [0, 1, 0, 1, 0],
        'ps_reg_01': [0.1, 0.2, 0.3, 0.4, 0.5],
        'ps_car_01_cat': [10, 11, 10, 11, 10]
    })
    
    print("Original data:")
    print(data)
    print(f"Shape: {data.shape}")
    
    # Encoder 생성 및 학습
    encoder = CategoricalEncoder()
    encoder.fit(data)
    
    # 변환
    transformed = encoder.transform(data)
    
    print("\nTransformed data:")
    print(transformed)
    print(f"Shape: {transformed.shape}")
    
    # 인코딩 정보 출력
    print("\nCategorical info:")
    print(encoder.get_categorical_info())
    
    return encoder, transformed


if __name__ == "__main__":
    test_categorical_encoder()