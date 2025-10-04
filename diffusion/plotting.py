import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

samples_from_cpp = False

if samples_from_cpp:
    samples = pd.read_csv("samples.csv", header=None)[0].to_numpy()
else:
    samples = torch.load("samples.pt").numpy()

def gaussian(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*((x-mu)/sigma)**2)

def mixture_density(x):
    return 0.5*gaussian(x, 0, 1) + 0.5*gaussian(x, 3, 1)

xs = np.linspace(-5, 8, 500)
ys = mixture_density(xs)

plt.figure(figsize=(8,5))
plt.hist(samples, bins=100, density=True, alpha=0.5, label="Samples")
plt.plot(xs, ys, "r-", linewidth=2, label="True density")
plt.legend()
plt.savefig("outputs/histogram.png")
plt.close()

