import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse

def print_clusters():
  csv_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
  df = pd.read_csv(csv_path,sep=",",header= None)

  df.columns = ["sepal_length","sepal_width","petal_length","petal_width","class"]

  df_kmeans = df[["sepal_length","sepal_width","petal_length","petal_width"]]

  k_means = KMeans(init = "k-means++", n_clusters = 5, n_init = 12)

  k_means.fit(df_kmeans)

  labels = k_means.labels_

  df["result"] = labels

  df.groupby('result').mean()

  # color =np.array(['red','blue','green','cyan','yellow'])
  # plt.scatter(x=df.sepal_length,y=df.sepal_width,c=color[k_means.labels_])
  # plt.title("Conjunto Iris")

  color =np.array(['red','blue','green','cyan','yellow'])
  scatter = plt.scatter(x=df.sepal_length,y=df.sepal_width,c=color[k_means.labels_])
  plt.title("Conjunto Iris")

  plt.savefig('iris.png')
  file = open('iris.png', mode="rb")

  return StreamingResponse(file, media_type="image/png")
