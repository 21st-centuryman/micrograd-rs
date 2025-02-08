import polars as pl
from sklearn.datasets import make_moons

x, y = make_moons(n_samples=100, noise=0.1)
y = y*2 - 1
df = pl.DataFrame({
    'x': x[:, 0],
    'y': x[:, 1],
    'label': y
})
df.write_csv('make_moons.csv')
