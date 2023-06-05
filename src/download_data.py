import urllib.request


url = "https://data.iowa.gov/api/views/38x4-vs5h/rows.csv"
filepath='./data/rows.csv'

urllib.request.urlretrieve(url, filepath)
