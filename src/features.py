from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class InteractionFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # fit 단계에서는 아무것도 학습할 필요가 없으므로 self를 그대로 반환
        return self

    def transform(self, X):
        X_new = X.copy()
        
        # 피처 엔지니어링
        
        return X_new

    def transform(self, X):
        X_new = X.copy()
        for col in self.categorical_features:
            # fit에서 만든 해당 피처의 인코딩 맵을 가져옴
            encoding_map = self.encoding_maps[col]
            
            # 매핑을 통해 새로운 피처들을 생성
            encoded_features = X_new[col].map(encoding_map.to_dict('index'))
            
            # 새로운 피처들의 이름 지정 (예: crop_type_target_enc_Urea)
            encoded_df = pd.DataFrame(encoded_features.tolist(), index=X_new.index)
            encoded_df.columns = [f'{col}_target_enc_{c}' for c in encoding_map.columns]
            
            # 기존 데이터프레임과 합치기
            X_new = pd.concat([X_new, encoded_df], axis=1)
        
        # 원본 범주형 컬럼은 이제 필요 없으니 삭제
        X_new = X_new.drop(columns=self.categorical_features)
        
        return X_new

class GroupStatsFeatureGenerator(BaseEstimator, TransformerMixin):
    # 이제 여러 개의 그룹과 여러 개의 수치형 컬럼을 처리하도록 리스트를 받는다
    def __init__(self, group_by_cols, numerical_cols):
        # group_by_cols: 어떤 컬럼으로 그룹을 묶을지 
        # numerical_cols: 어떤 컬럼들의 통계량을 계산할지
        self.group_by_cols = group_by_cols
        self.numerical_cols = numerical_cols
        self.stats_df = None

    def fit(self, X, y=None):
        # 파이프라인이 .fit()을 호출할 때 실행
        # 훈련 데이터 X를 받아 피처를 만들기 위한 '규칙'(여기서는 '그룹별 통계량')을 학습
        # 훈련 데이터만으로 통계량을 계산하고, 나중에 변환을 위해 저장
        data = X.copy()
        
        self.stats_df = data.groupby(self.group_by_cols)[self.numerical_cols].agg(['mean', 'std']).reset_index()
        # __init__에서 기억해둔 그룹 기준 컬럼으로 데이터를 그룹화
        # 통계량을 계산할 수치형 컬럼들(예: ['Nitrogen', 'Potassium'])을 선택
        # 선택된 수치형 컬럼들에 대해 평균(mean)과 표준편차(std)를 계산
        # .reset_index(): groupby 결과를 다시 일반적인 데이터프레임 형태로 변환
        # 컬럼 이름 재설정 로직도 거의 그대로 사용 가능
        # self.stats_df: 이렇게 계산된 최종 통계표를, 
        # 아까 만들어둔 '빈 보관함' self.stats_df에 저장
        
        # Flattening
        group_by_str = '_'.join(self.group_by_cols)
        # 그룹 컬럼 이름을 합쳐서 사용
        new_cols = []
        for col in self.stats_df.columns:
            if col[1]: # 멀티인덱스의 두 번째 레벨 이름이 존재하면 (mean, std) 
                new_cols.append(f'{col[0]}_{col[1]}_by_{group_by_str}')
            else: # 멀티인덱스의 두 번째 레벨 이름이 없으면 (group_by_cols)
                new_cols.append(col[0])
        self.stats_df.columns = new_cols
        
        return self

    def transform(self, X):
        # .fit()에서 학습한 규칙을 사용해 이 데이터를
        X_new = X.copy()
        
        # fit에서 계산해둔 통계량을 원본 데이터에 병합(merge)
        X_new = pd.merge(X_new, self.stats_df, on=self.group_by_cols, how='left')
        # merge: 두개의 df 결합
        # X_new, self.stats_df: 병합할 데이터
        # on=self.group_by_cols: 
        # 그룹 기준 컬럼(예: Crop Type)을 키(key)로 사용해서 두 테이블을 병합
        # how='left': 왼쪽 테이블(X_new)의 모든 행을 유지
        
        return X_new
        # 변환 작업이 끝난, 새로운 피처들이 추가된 최종 데이터프레임을 반드시 반환해야 함. 
        # 이 결과물이 파이프라인의 다음 '부품'으로 전달됨.
        