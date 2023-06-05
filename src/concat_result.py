import glob

import pandas as pd


def concat_results() -> None:
    file_list = glob.glob('./result/result*.txt')

    # 空の DataFrame を作成
    df = pd.DataFrame()

    # ファイルごとにループ処理
    for file in file_list:
        # JSON ファイルを読み込んで DataFrame に追加
        data = pd.read_json(file).T
        data['file'] = file
        df = pd.concat([df, data], axis=0)

    df.to_csv('./result/result.csv')


if __name__ == '__main__':
    concat_results()
