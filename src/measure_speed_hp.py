import argparse
import json
import time

import polars as pl
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.svm import SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, NMF
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from loguru import logger
logger.add("run_all.log") 


MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "BayesianRidge": BayesianRidge(),
    "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=7),
    "RandomForestRegressor": RandomForestRegressor(max_depth=7),
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "SVR": SVR(), # kernel=linearだと遅い
    "LGBMRegressor": LGBMRegressor(),
    "XGBRegressor": XGBRegressor(),
    "CatBoostRegressor": CatBoostRegressor(),
    "KMeans": KMeans(),
    "GaussianMixture": GaussianMixture(),
    "DBSCAN": DBSCAN(),
    "PCA": PCA(),
    "NMF": NMF(),
    "NearestNeighbors": NearestNeighbors(),
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('n')
    parser.add_argument('algo')
    return parser.parse_args()


def fit_and_predict(model, X, y) -> dict:
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    fit_time = end_time - start_time

    try:
        start_time = time.time()
        model.predict(X)
        end_time = time.time()
        predict_time = end_time - start_time
    except:
        predict_time = None

    return {'model': model.__class__.__name__, 'fit': fit_time, 'predict':predict_time}


def main():
    args = get_args()
    samplesize = int(args.n)
    algorithm = args.algo

    df = pl.scan_csv('./data/rows.csv').fetch(samplesize)

    if df.height < samplesize:
        df = df.sample(samplesize, with_replacement=True)
    logger.info(f'df shape: {df.shape}')

    X_nonnull = df.select([
        pl.col(pl.Int64), pl.col(pl.Float64)
    ]).select(
        pl.exclude('Sale (Dollars)')
    ).fill_nan(0).fill_null(0)
    y = df['Sale (Dollars)'].fill_nan(0).fill_null(0)

    model = MODELS[algorithm]

    if algorithm == 'CatBoostRegressor':
        # cannot use polars for catboost; transform to pandas
        X_nonnull = X_nonnull.to_pandas()
        y = y.to_pandas()
        model = CatBoostRegressor(verbose=0)

    _ = fit_and_predict(model, X_nonnull, y)
    print(f'"{model.__class__.__name__}":')
    print(json.dumps(_, indent=4), ',')

if __name__ == "__main__":
    main()
