import jax

jax.config.update("jax_enable_x64", True)

from Lintel import intel
from Lintel.gp_utils import GP
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


s = "data/noumenta/realKnownCause/realKnownCause/ambient_temperature_system_failure.csv"
train_y = pd.read_csv(s).value.to_numpy()
train_t = np.arange(len(train_y))

pretrain_t = train_t[:100]
train_t = train_t[100:1000]

pretrain_y = train_y[:100]
train_y = train_y[100:1000]

gp1 = GP(
    np.array(pretrain_t),
    np.array(pretrain_y),
    lengthscale=10.0,
    sigma_f=1.0,
    sigma_n=pretrain_y.std(),
    C=pretrain_y.mean(),
)

gp1.maximize_evidence(pretrain_t, pretrain_y, lr=1e-2, iters=5000, verbose=False)

gp2 = GP(
    np.array(pretrain_t),
    np.array(pretrain_y),
    lengthscale=gp1.lengthscale,
    sigma_f=gp1.sigma_f / 5.0,
    sigma_n=5.0 * gp1.sigma_n,
    C=pretrain_y.mean(),
)

gp3 = GP(
    np.array(pretrain_t),
    np.array(pretrain_y),
    lengthscale=gp1.lengthscale,
    sigma_f=gp1.sigma_f,
    sigma_n=gp1.sigma_n / 5.0,
    C=pretrain_y.mean(),
)

gp4 = GP(
    np.array(pretrain_t),
    np.array(pretrain_y),
    lengthscale=gp1.lengthscale,
    sigma_f=gp1.sigma_f / 5.0,
    sigma_n=gp1.sigma_n / 2.0,
    C=pretrain_y.mean(),
)

intel = intel.INTEL(
    N=3,
    tau=20,
    alpha=0.9,
    weights=np.ones(
        4,
    )
    / 4,
    gps=[gp1, gp2, gp3, gp4],
    L=10,
)

ms = []
ss = []
ots = []
ws = []

for t in tqdm(range(len(train_y))):
    m, s, o = intel.predict_and_update(train_t[t], train_y[t])
    ws.append(intel.weights)

    if o:
        ots.append(t)

    ms.append(m)
    ss.append(s)

ms = np.array(ms)
ss = np.array(ss)

plt.scatter(pretrain_t, pretrain_y)
plt.scatter(train_t, train_y)
plt.plot(train_t, ms, color="black")
plt.fill_between(
    train_t, ms + 2 * np.sqrt(ss), ms - 2 * np.sqrt(ss), color="black", alpha=0.3
)
for ot in ots:
    plt.scatter(train_t[ot], train_y[ot], color="green", marker="+", s=250)

plt.savefig("Intel_Test.png")

plt.clf()

plt.scatter(train_t[500:600], train_y[500:600])
plt.plot(train_t[500:600], ms[500:600], color="black")
plt.fill_between(
    train_t[500:600],
    ms[500:600] + 2 * np.sqrt(ss)[500:600],
    ms[500:600] - 2 * np.sqrt(ss)[500:600],
    color="black",
    alpha=0.3,
)
for ot in ots:
    # plt.axvline(train_t[ot])
    if ot < 500 or ot > 600:
        continue
    plt.scatter(train_t[ot], train_y[ot], color="green", marker="+", s=250)

plt.savefig("Intel_Test_Closeup.png")
