from pathlib import Path
import pandas as pd 
import numpy as np
import pickle
import sys
from lightgbm import LGBMRegressor
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", message="[LightGBM]*")

import logging
import lightgbm as lgb

logger = logging.getLogger('lightgbm')
logger.setLevel(logging.ERROR)
logger.setLevel(logging.NOTSET)




root_dir = Path('.')
data_dir = root_dir / 'data'
result_dir = root_dir / 'Results'
fold_num = sys.argv[1]
fold_dir = result_dir / f'fold_{fold_num}'
fold_data_dir = fold_dir / 'data'
fold_data_path = fold_data_dir / 'result_emb.pkl'

with open(fold_data_path, 'rb') as f:
    X_train, y_train, X_valid, y_valid, X_test, y_test = pickle.load(f)


lgbm_search_space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'max_depth': hp.choice('max_depth', np.arange(3, 10, dtype=int)),
    'n_estimators': hp.choice('n_estimators', np.arange(100, 1000, 100, dtype=int)),
    'num_leaves': hp.choice('num_leaves', np.arange(10, 100, 10, dtype=int)),
    'min_child_samples': hp.choice('min_child_samples', np.arange(10, 100, 10, dtype=int)),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
}

trials = Trials()
def objective(params):
    params = {
        'learning_rate': params['learning_rate'],
        'max_depth': params['max_depth'],
        'n_estimators': params['n_estimators'],
        'num_leaves': params['num_leaves'],
        'min_child_samples': params['min_child_samples'],
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree'],
    }
    model = LGBMRegressor(**params, objective='mae')
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    mae = model.best_score_['valid_0']['l1']
    return mae
    

best = fmin(fn=objective, space=lgbm_search_space, algo=tpe.suggest, max_evals=100, trials=trials, verbose=0)
model = LGBMRegressor(**best)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
tpe_results = pd.DataFrame(trials.results)
print(tpe_results)