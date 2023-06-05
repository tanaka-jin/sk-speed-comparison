import argparse

import pandas as pd
import plotly.graph_objects as go


def format_dataframe_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    df['size'] = df.file.str.extract(r'(\d+)').astype(int)
    df['hp'] = df.file.str.contains('hp')

    df['model_type'] = df['model'].replace({
        'LinearRegression': 'linear', 
        'Ridge': 'linear',
        'Lasso': 'linear', 
        'BayesianRidge': 'linear',
        'DecisionTreeRegressor': 'tree', 
        'RandomForestRegressor': 'tree',
        'GradientBoostingRegressor': 'gbdt', 
        'LGBMRegressor': 'gbdt', 
        'XGBRegressor': 'gbdt', 
        'CatBoostRegressor': 'gbdt', 
        'KNeighborsRegressor': 'clustering',
        'KMeans': 'clustering',
        'GaussianMixture': 'clustering', 
        'DBSCAN': 'clustering', 
        'NearestNeighbors': 'clustering',
        'PCA': 'dim reduction', 
        'NMF': 'dim reduction', 
        'TSNE': 'dim reduction', 
        'SVR': 'other', 
        'GaussianProcessRegressor': 'other'
    })

    df['color'] = df['model_type'].replace({
        'linear': '#636EFA',
        'tree': '#EF553B',
        'gbdt': '#00CC96',
        'clustering': '#AB63FA',
        'dim reduction': '#FFA15A',
        'other': '#19D3F3'
    })

    df['model_type_id'] = df['model_type'].replace({
        'linear': 1,
        'tree': 2,
        'gbdt': 3,
        'clustering': 4,
        'dim reduction': 5,
        'other': 6
    })


    df['model_id'] = df['model'].replace({
        'LinearRegression': '1', 
        'Ridge': '2',
        'Lasso': '3', 
        'BayesianRidge': '4',
        'DecisionTreeRegressor': '1', 
        'RandomForestRegressor': '2',
        'GradientBoostingRegressor': '1', 
        'LGBMRegressor': '2', 
        'XGBRegressor': '3', 
        'CatBoostRegressor': '4', 
        'KNeighborsRegressor': '1',
        'KMeans': '2',
        'GaussianMixture': '3', 
        'DBSCAN': '4',
        'NearestNeighbors': '5',
        'PCA': '1', 
        'NMF': '2', 
        'TSNE': '3', 
        'SVR': '1', 
        'GaussianProcessRegressor': '2'
    })

    df['linetype'] = df['model_id'].replace({
        '1': 'solid',
        '2': 'dash',
        '3': 'dot',
        '4': 'dashdot'
    })

    df = df.sort_values(['size', 'model_type_id', 'model_id'])
    return df


def plot_by_size(df: pd.DataFrame) -> None:
    traces = []
    for g, df_g in df.query('size!=10001').groupby('model', sort=False):
        trace = go.Scatter(
            x=df_g['size'],
            y=df_g['fit'] + 1e-2,
            mode='markers + lines',
            text=df_g['model'],
            marker=dict(
                size=7,
                color=df_g['color']
            ),
            line=dict(
                color=df_g['color'].unique()[0],
                dash=df_g['linetype'].unique()[0]
            ),
            name=g
        )
        traces.append(trace)

    fig = go.Figure(traces)
    fig.update_layout(
        xaxis_type='log', 
        yaxis_type='log',
        width=900, 
        height=700,
        template='plotly_white',
        xaxis_title='Sample Size',
        yaxis_title='Time (seconds)',
        font=dict(size=16)
    )

    # save
    fig.write_html('./graph/time_by_size.html')
    fig.write_image('./graph/time_by_size.jpg', scale=4)
    fig.write_image('./graph/time_by_size.pdf')

    # normal(non-log) scale
    fig.update_layout(
        xaxis_type='linear', 
        yaxis_type='linear',
    )
    fig.write_html('./graph/time_by_size_lin.html')
    fig.write_image('./graph/time_by_size_lin.jpg', scale=4)
    fig.write_image('./graph/time_by_size_lin.pdf')



def plot_by_model(df: pd.DataFrame, n: int) -> None:
    data = df[(df['size'] == n)].copy()
    data['model'] = data['model'] + ' '
    trace1 = go.Bar(y=data['model'], x=data['fit'], name='fit', 
                    orientation='h', marker_color='grey')
    trace2 = go.Bar(y=data['model'], x=data['predict'], name='predict', 
                    orientation='h', marker_color='moccasin')

    fig = go.Figure([trace1, trace2])
    fig.update_layout(
        xaxis_type='log', 
        width=900, 
        height=700,
        template='plotly_white',
        yaxis=dict(autorange="reversed"),
        legend=dict(xanchor='right', yanchor='top'),
        xaxis_title='Time (seconds)',
        font=dict(size=18)
    )

    fig.write_image(f'./graph/time_{n}.jpg', scale=4)
    fig.write_image(f'./graph/time_{n}.pdf')

    # normal(non-log) scale
    fig.update_layout(
        xaxis_type='linear', 
    )
    fig.write_image(f'./graph/time_{n}_lin.jpg', scale=4)
    fig.write_image(f'./graph/time_{n}_lin.pdf')


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='./result/result.csv')
    args = parser.parse_args()
    file_path = args.file

    df = pd.read_csv(file_path)
    df = format_dataframe_for_plot(df)

    plot_by_size(df)
    plot_by_model(df, 3_000)
    plot_by_model(df, 1_000_000)
    plot_by_model(df, 10_000_000)

    
