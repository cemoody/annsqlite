This repo builds wraps simple nearest-neighbor searching within a SQLite DB
and exposes it via datasette.

Download glove-25: http://ann-benchmarks.com/glove-25.hdf5

Follow directions to build JSON extension: https://stackoverflow.com/questions/39319280/python-sqlite-json1-load-extension 


virtual table method:

select * 
from embed
where k=20, metric='dot',pc1=-1.2,...pc32=3.2
