import time
import json
import h5py
import random
import sqlite3
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
# from joblib import Memory
# mem = Memory('cache')

n = 3
cnt = 0


def get_data(fn="glove-25-angular.hdf5"):
    dataset = h5py.File(fn, 'r')
    train = dataset['train']
    df = pd.DataFrame(train)
    return df


def pyrand():
    return random.random()


def build_index(labels, vectors, n_trees=16):
    t0 = time.time()
    n = vectors.shape[0]
    dim = vectors.shape[1]
    p = AnnoyIndex(dim, 'dot')
    for label, vector in zip(labels, vectors):
        p.add_item(label, vector)
    p.build(n_trees, n_jobs=-1)
    t1 = time.time()
    print(f"built index in {t1-t0: 1.3f}s")
    return p


def knn_query(k, vector):
    global p
    labels, distances = p.get_nns_by_vector(vector, k, search_k=-1, include_distances=True)
    labels, distances = list(labels), list(distances)
    return json.dumps(dict(labels=labels, distances=distances))


# Initialize DB
con = sqlite3.connect(':memory:')
# con.enable_load_extension(True)
# con.execute("select load_extension('json1')")


# hosting in memory subtracts 200ms from queries
df = get_data()
labels = df.index
vectors = df.values
dimension = vectors.shape[1]
p = build_index(labels, vectors)
dist_cache = {}
t0 = time.time()
df2 = df.copy()
df2['id'] = labels
df2.to_sql('embed', con)
t1 = time.time()
cur = con.cursor()
print(f"Insert embed in {t1-t0:1.3f}")

for _ in range(3):
    t0 = time.time()
    choice = random.choice(labels)
    query = np.random.randn(dimension)
    blob = knn_query(20, query)
    t1 = time.time()
    blob = knn_query(20, query)
    t2 = time.time()
    print(f"Time to do initial search was {t1-t0:1.3f}s")
    print(f"Time to do second search was {t2-t1:1.3f}s")

# try out HNSW within sqlite
con.create_function("knn_query", 1 + dimension, knn_query)
query_str = ','.join(str(x) for x in np.random.randn(dimension))
for i in range(n):
    t0 = time.time()
    cur.execute(f"select * from embed " +
                f"where id in knn_query(20, {query_str})")
    x = cur.fetchall()
    t1 = time.time()
    print(f"Time to do knn query is {t1-t0:1.3f}s")



# takes ~250-300ms on disk, but 50ms in mem
# can take up to 90ms with extra function overhead
for _ in range(n):
    t0 = time.time()
    cur.execute("select * from embed order by pyrand() limit 100")
    x = cur.fetchall()
    t1 = time.time()
    dt = t1 - t0
    print(f"Time to do pyrand is {t1-t0:1.3f}s")
