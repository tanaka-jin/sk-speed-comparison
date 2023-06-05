import os
import subprocess

from loguru import logger
logger.add("run_all.log") 


ALGORITHMS = [
    "LinearRegression", 
    "Ridge", 
    "Lasso", 
    "BayesianRidge", 
    "DecisionTreeRegressor", 
    "RandomForestRegressor", 
    "GradientBoostingRegressor", 
    "KNeighborsRegressor", 
    "SVR", 
    "LGBMRegressor", 
    "XGBRegressor", 
    "CatBoostRegressor", 
    "KMeans", 
    "GaussianMixture", 
    "DBSCAN", 
    "PCA", 
    "NMF", 
    "NearestNeighbors", 
    "TSNE", 
    "GaussianProcessRegressor", 
]



def run_subprocess(n, algorithm):
    # コマンドの実行
    logger.info(f'    start {algorithm}')

    if n <= 10_000:
        cmd = f'python ./src/measure_speed.py {n} {algorithm}'
    else:
        cmd = f'python ./src/measure_speed_hp.py {n} {algorithm}'
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=600,
            shell=True, 
            encoding="shift-jis"
        )
        # 標準出力の取得
        return result.stdout

    except subprocess.TimeoutExpired:
        logger.info(f'    {algorithm} timeout')




def main(n):
    logger.info(f'### START {n} ###')
    file_path = f'./result/result{n}.txt'


    # 初期化
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass

    with open(file_path, "w") as file:
        file.write("{")

    # アルゴリズムごとに実行
    for algorithm in ALGORITHMS:
        result = run_subprocess(n, algorithm)
        if not result:
            continue

        with open(file_path, "a") as file:
            file.write(result)

    with open(file_path, "a") as file:
        file.write("}")

    logger.info(f'### END {n} ###')


if __name__ == "__main__":
    logger.info('### START run_all ###')
    for i in [100, 300, 1000, 3000, 10000, 10_001, 30_000, 
              100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]:
        main(i)
