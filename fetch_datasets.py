import io
import zipfile

import requests

# Download the latest (small) dataset
url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))

# Unzip the dataset into a folder
z.extractall("data/")

# Download the latest (full) dataset
url = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))

# Unzip the dataset into a folder
z.extractall("data/")
