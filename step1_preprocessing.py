from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

root_dir = Path('.')
data_dir = root_dir / 'data'
result_dir = root_dir / 'Results'

train_df_path = data_dir / 'train.csv'
test_df_path  = data_dir / 'test.csv'

index_col = "SAMPLE_ID"
target_col = "CI_HOUR"
# NA값들 처리
na_cols = ["U_WIND", "V_WIND", 'AIR_TEMPERATURE', 'BN'] # 아예 제거
subset_cols = ['BREADTH', 'DEPTH', 'DRAUGHT', 'LENGTH'] # NA가 있는 행들 제거

# Encoding
str_cols = [ # Label encoding 변환을 위하여 선정
    "ARI_CO", # 도착항 소속 국가
    "ARI_PO", # 도착항 항구 명
    "SHIP_TYPE_CATEGORY",  # 5대 선종
    "ID", # 선박 ID
    "SHIPMANAGER", # 선박 소유주
    "FLAG", # 선박 국적
    "BREADTH", # 선박 폭
    "DEPTH", # 선박 깊이
    "DRAUGHT", # 선박 최대 허깨비
    "ATA_LT", # 도착항 시간F
    'BUILT', # 선박 나이
    'GT', # 선박 총톤수
    ###################### 전처리로 추가한 변수들 ######################
    "year", # 연도
    "month", # 월
    "weekday", # 요일
    "morning", # 오전 여부
]

num_cols = [
    'DEADWEIGHT',
    'PORT_SIZE',
    'DIST',
]

drop_cols = [
    'SAMPLE_ID', 
    'CI_HOUR', 
    'ATA',
    ]


def date_preprocessing(df) :
    df['ATA'] = pd.to_datetime(df['ATA'])
    df['year'] = df['ATA'].dt.year
    df['month'] = df['ATA'].dt.month
    df['weekday'] = df['ATA'].dt.weekday
    df['hour'] = df['ATA'].dt.hour
    df['morning'] = df['hour'].apply(lambda x : 1 if x < 12 else 0)
    df.drop(['ATA', 'hour'], axis = 1, inplace = True)
    return df

def load_df(data_path) : 
    df = pd.read_csv(data_path)
    df = date_preprocessing(df)
    return df

###################### Load Data ######################
## Datetime Preprocessed : YYYY-MM-DD HH:MM:SS -> YYYY, MM, Weekday
train_df = load_df(train_df_path)
test_df  = load_df(test_df_path)


train_df = train_df.dropna(subset = subset_cols, axis = 0)
train_df = train_df.drop(na_cols, axis = 1)
test_df = test_df.drop(na_cols, axis = 1)

# ###################### Stratified K-Fold ######################
train_label = train_df[target_col].to_numpy()
train_binary = np.where(train_label == 24, 0, 1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold_num, (train_idx, valid_idx) in enumerate(skf.split(train_df, train_binary)) : 
    fold_data_dir = result_dir.joinpath(f"fold_{fold_num}", "data")
    fold_data_dir.mkdir(exist_ok = True, parents = True)
    
    fold_train = train_df.iloc[train_idx]
    fold_valid = train_df.iloc[valid_idx]
    fold_test = test_df.copy()
    
    for str_col in str_cols : 
        le = {x : idx for idx, x in enumerate(sorted(fold_train[str_col].unique()), 1)} # Label Encoding
        le['UNK'] = 0
        fold_train[str_col] = fold_train[str_col].map(le).fillna(0).astype(int)
        fold_valid[str_col] = fold_valid[str_col].map(le).fillna(0).astype(int)
        fold_test[str_col] = fold_test[str_col].map(le).fillna(0).astype(int)
    
    ss = StandardScaler()
    ss.fit(fold_train[num_cols])
    fold_train[num_cols] = ss.transform(fold_train[num_cols])
    fold_valid[num_cols] = ss.transform(fold_valid[num_cols])
    fold_test[num_cols] = ss.transform(fold_test[num_cols])
    
    fold_train.to_csv(fold_data_dir.joinpath('train.csv'), index = False)
    fold_valid.to_csv(fold_data_dir.joinpath('valid.csv'), index = False)
    fold_test.to_csv( fold_data_dir.joinpath('test.csv'), index = False)
    
    print(f"Fold {fold_num} is Done! | Train : {fold_train.shape} | Valid : {fold_valid.shape} | Test : {fold_test.shape}")


"""
import pandas as pd
df = pd.read_csv('data/train_fold_df.csv')
print(df.isna().sum())
"""