import matplotlib.pyplot as plt
import pandas as pd

samples = pd.read_csv("samples.csv", header=None)[0]
plt.hist(samples, bins=100, density=True)
plt.savefig("histogram.png")