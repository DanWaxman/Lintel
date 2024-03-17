import jax

jax.config.update("jax_enable_x64", True)

from Lintel import lintel, intel
from Lintel.gp_utils import GP, MarkovianGP, SubbandMatern32
import bayesnewton
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import time

parser = argparse.ArgumentParser(description="Example 1")
parser.add_argument("--fit_mode", nargs="?", default=False, type=bool)
parser.add_argument("--plot_mode", nargs="?", default=False, type=bool)
parser.add_argument("--seed", nargs="?", default=0, type=int)
parser.add_argument("--geometric_fusion", nargs="?", default=False, type=bool)

args = parser.parse_args()

sns.set_theme(context="paper", style="whitegrid", palette="colorblind")
LINTEL_COLOR = "orange"
INTEL_COLOR = "blue"

FIT_MODE = args.fit_mode
PLOT_MODE = args.plot_mode

SEED = args.seed

print("####################")
print(f"Running with seed {SEED}")

if FIT_MODE:
    np.random.seed(SEED)

    print("Generating Data")
    t = np.sort(3000 * np.random.uniform(size=3000)).reshape(-1, 1)
    k_truth_1 = bayesnewton.kernels.Sum(
        [
            SubbandMatern32(
                radial_frequency=1.0 / 50.0, lengthscale=500.0, variance=1.5
            ),
            SubbandMatern32(
                radial_frequency=1.0 / 20.0, lengthscale=300.0, variance=0.7
            ),
            SubbandMatern32(
                radial_frequency=1.0 / 15.0, lengthscale=100.0, variance=0.2
            ),
        ]
    )

    y = np.zeros_like(t).squeeze()
    y = np.random.multivariate_normal(
        mean=np.zeros((3000,)), cov=k_truth_1.K(t[:3000], t[:3000])
    )

    y = y + np.random.randn(*y.shape) * 0.3
    true_outliers = -np.random.randint(0, 1750, size=10)
    for outlier in true_outliers:
        y[outlier] = y[outlier] + np.random.randn() * 2.0

    pretrain_t = t[:250].reshape(-1, 1)
    train_t = t[250:].reshape(-1, 1)

    pretrain_y = y[:250]
    train_y = y[250:]
    print("Finished Generating Data")

    np.savez(
        f"results/experiment_1_seed_{SEED}_geom_{args.geometric_fusion}_data.npz",
        pretrain_t=pretrain_t,
        train_t=train_t,
        pretrain_y=pretrain_y,
        train_y=train_y,
        true_outliers=true_outliers,
    )

    k0 = bayesnewton.kernels.Sum(
        [
            SubbandMatern32(),
            SubbandMatern32(),
            SubbandMatern32(),
        ]
    )

    gp0 = GP(
        np.array(pretrain_t),
        np.array(pretrain_y),
        C=pretrain_y.mean(),
        sigma_n=0.1,
        kernel=k0,
    )

    print("Pretraining GP")
    gp0.maximize_evidence(pretrain_t, pretrain_y, lr=1e-1, iters=2500, verbose=True)
    print("Finished Pretraining")

    k1 = k0
    k2 = bayesnewton.kernels.Matern52(lengthscale=50.0, variance=0.2)

    gp1 = MarkovianGP(C=pretrain_y.mean(), sigma_n=gp0.sigma_n, kernel=k1)
    gp1.reset_and_filter(pretrain_t.squeeze()[-1:], pretrain_y[-1:], pretrain_y.mean())

    gp2 = MarkovianGP(C=pretrain_y.mean(), sigma_n=gp0.sigma_n, kernel=k2)
    gp2.reset_and_filter(pretrain_t.squeeze()[-1:], pretrain_y[-1:], pretrain_y.mean())

    print("##################")
    print("#### LINTEL #####")
    print("##################")
    lintel = lintel.LINTEL(
        N=3,
        alpha=0.9,
        weights=np.ones(
            2,
        )
        / 2,
        gps=[gp1, gp2],
        L=10000,
        verbose=True,
        product_of_experts=args.geometric_fusion,
    )

    ms = []
    ss = []
    ots = []
    ws = []

    t_start = time.time()
    for t in tqdm(range(len(train_y))):
        m, s, o = lintel.predict_and_update(train_t[t, 0], train_y[t])
        ws.append(lintel.weights)

        if o:
            ots.append(t)

        ms.append(m)
        ss.append(s)

    t_end = time.time()
    t_lintel = t_end - t_start
    ms = np.array(ms)
    ss = np.array(ss)
    ots = np.array(ots)
    ws = np.array(ws)

    np.savez(
        f"results/experiment_1_seed_{SEED}_geom_{args.geometric_fusion}_lintel_results.npz",
        ms=ms,
        ss=ss,
        ots=ots,
        ws=ws,
        t_lintel=t_lintel,
    )

    gp1 = GP(
        pretrain_t, pretrain_y, C=pretrain_y.mean(), sigma_n=gp0.sigma_n, kernel=k1
    )

    gp2 = GP(
        pretrain_t,
        pretrain_y,
        C=pretrain_y.mean(),
        sigma_n=gp0.sigma_n,
        kernel=k2,
    )

    print("#################")
    print("#### INTEL #####")
    print("#################")
    intel = intel.INTEL(
        N=3,
        tau=20,
        alpha=0.9,
        weights=np.ones(
            2,
        )
        / 2,
        gps=[gp1, gp2],
        L=10000,
        verbose=True,
        product_of_experts=args.geometric_fusion,
    )

    ms = []
    ss = []
    ots = []
    ws = []

    t_start = time.time()
    for t in tqdm(range(len(train_y))):
        m, s, o = intel.predict_and_update(train_t[t], train_y[t])
        ws.append(intel.weights)

        if o:
            ots.append(t)

        ms.append(m)
        ss.append(s)

    t_end = time.time()
    t_intel = t_end - t_start

    ms = np.array(ms)
    ss = np.array(ss)
    ots = np.array(ots)
    ws = np.array(ws)

    np.savez(
        f"results/experiment_1_seed_{SEED}_geom_{args.geometric_fusion}_intel_results.npz",
        ms=ms,
        ss=ss,
        ots=ots,
        ws=ws,
        t_intel=t_intel,
    )

### Plotting
if PLOT_MODE:
    data = np.load(
        f"results/experiment_1_seed_{SEED}_geom_{args.geometric_fusion}_data.npz"
    )
    train_t = data["train_t"].squeeze()
    pretrain_t = data["pretrain_t"].squeeze()
    train_y = data["train_y"]
    pretrain_y = data["pretrain_y"]
    true_outliers = data["true_outliers"]

    res_lintel = np.load(
        f"results/experiment_1_seed_{SEED}_geom_{args.geometric_fusion}_lintel_results.npz"
    )
    m_lintel = res_lintel["ms"]
    s_lintel = res_lintel["ss"]
    o_lintel = res_lintel["ots"]
    w_lintel = res_lintel["ws"]

    res_intel = np.load(
        f"results/experiment_1_seed_{SEED}_geom_{args.geometric_fusion}_intel_results.npz"
    )
    m_intel = res_intel["ms"]
    s_intel = res_intel["ss"]
    o_intel = res_intel["ots"]
    w_intel = res_intel["ws"]

    def moving_average(x, w=10):
        return np.convolve(x, np.ones(w), "valid") / w

    intel_pll = stats.norm.logpdf(train_y, m_intel, np.sqrt(s_intel))
    lintel_pll = stats.norm.logpdf(train_y, m_lintel, np.sqrt(s_lintel))
    print(f"MPLL for INTEL: {np.delete(intel_pll, true_outliers).mean()}")
    print(f"MPLL for LINTEL: {np.delete(lintel_pll, true_outliers).mean()}")

    intel_nmse = (train_y - m_intel) ** 2 / np.var(train_y)
    lintel_nmse = (train_y - m_lintel) ** 2 / np.var(train_y)
    print(f"nMSE for INTEL: {np.delete(intel_nmse, true_outliers).mean()}")
    print(f"nMSE for LINTEL: {np.delete(lintel_nmse, true_outliers).mean()}")

    plt.scatter(range(len(moving_average(intel_pll))), moving_average(intel_pll))
    plt.scatter(range(len(moving_average(lintel_pll))), moving_average(lintel_pll))
    plt.savefig("plots/experiment_1_plls.png")
    plt.clf()
    print(train_t.shape, intel_pll.shape)
    plt.scatter(train_t, intel_pll - lintel_pll)
    plt.scatter(train_t[true_outliers], (intel_pll - lintel_pll)[true_outliers])
    plt.savefig("plots/experiment_1_pll_diffs.png")

    fig, axs = plt.subplots(
        nrows=2, figsize=(7.5, 2), sharex=True, height_ratios=[2, 1]
    )
    ax1, ax2 = axs

    ax1.scatter(pretrain_t, pretrain_y, alpha=0.75, s=2, label="Pretraining Points")
    ax1.scatter(
        train_t, train_y, color="black", alpha=0.75, s=2, label="Training Points"
    )
    ax1.scatter(
        train_t[o_lintel],
        train_y[o_lintel],
        marker="x",
        color="red",
        alpha=0.8,
        label="LINTEL Outlier",
    )
    ax1.scatter(
        train_t[o_intel],
        train_y[o_intel],
        marker="+",
        color="red",
        alpha=0.8,
        label="INTEL Outlier",
    )
    ax1.plot(train_t, m_lintel, label="LINTEL", color=LINTEL_COLOR)
    ax1.fill_between(
        train_t,
        m_lintel + 2 * np.sqrt(s_lintel),
        m_lintel - 2 * np.sqrt(s_lintel),
        alpha=0.3,
        color=LINTEL_COLOR,
    )
    ax1.plot(train_t, m_intel, linestyle="--", label="INTEL", color=INTEL_COLOR)
    ax1.fill_between(
        train_t,
        m_intel + 2 * np.sqrt(s_intel),
        m_intel - 2 * np.sqrt(s_intel),
        alpha=0.3,
        color=INTEL_COLOR,
    )

    zoom_start = 1800
    zoom_end = 2000
    x1, x2, y1, y2 = (
        train_t[zoom_start],
        train_t[zoom_end],
        np.min(
            m_lintel[zoom_start:zoom_end] - 2 * np.sqrt(s_lintel)[zoom_start:zoom_end]
        )
        - 1.5,
        np.max(
            m_lintel[zoom_start:zoom_end] + 2 * np.sqrt(s_lintel)[zoom_start:zoom_end]
        )
        + 1.5,
    )
    axin1 = ax1.inset_axes(
        [0.8, 0.65, 0.15, 0.3],
        xlim=(x1, x2),
        ylim=(y1, y2),
        xticks=[],
        yticks=[],
        xticklabels=[],
        yticklabels=[],
    )

    axin1.scatter(
        train_t[zoom_start:zoom_end],
        train_y[zoom_start:zoom_end],
        color="black",
        alpha=0.75,
        s=2,
    )
    axin1.plot(
        train_t[zoom_start:zoom_end], m_lintel[zoom_start:zoom_end], color=LINTEL_COLOR
    )
    axin1.fill_between(
        train_t[zoom_start:zoom_end],
        m_lintel[zoom_start:zoom_end] + 2 * np.sqrt(s_lintel)[zoom_start:zoom_end],
        m_lintel[zoom_start:zoom_end] - 2 * np.sqrt(s_lintel)[zoom_start:zoom_end],
        alpha=0.3,
        color=LINTEL_COLOR,
    )
    axin1.plot(
        train_t[zoom_start:zoom_end],
        m_intel[zoom_start:zoom_end],
        linestyle="--",
        color=INTEL_COLOR,
    )
    axin1.fill_between(
        train_t[zoom_start:zoom_end],
        m_intel[zoom_start:zoom_end] + 2 * np.sqrt(s_intel)[zoom_start:zoom_end],
        m_intel[zoom_start:zoom_end] - 2 * np.sqrt(s_intel)[zoom_start:zoom_end],
        alpha=0.3,
        color=INTEL_COLOR,
    )
    os = o_lintel[(o_lintel > zoom_start) * (o_lintel < zoom_end)]
    axin1.scatter(
        train_t[os],
        train_y[os],
        marker="x",
        color="red",
        alpha=0.8,
        label="LINTEL Outlier",
    )
    os = o_intel[(o_intel > zoom_start) * (o_intel < zoom_end)]
    axin1.scatter(
        train_t[os],
        train_y[os],
        marker="+",
        color="red",
        alpha=0.8,
        label="INTEL Outlier",
    )

    ax1.indicate_inset_zoom(axin1, edgecolor="black", alpha=1, lw=0.7)

    # Make legend. It's easier to bound it to ax2 if we want it at the bottom
    ax2.scatter([], [], alpha=0.75, s=2, label="Pretraining Points")
    ax2.scatter([], [], color="black", alpha=0.75, s=2, label="Training Points")
    ax2.scatter([], [], color="red", marker="x", alpha=0.8, label="LINTEL Outlier")
    ax2.scatter([], [], color="red", marker="+", alpha=0.8, label="INTEL Outlier")
    ax2.plot([], [], color=LINTEL_COLOR, label="LINTEL")
    ax2.plot([], [], color=INTEL_COLOR, label="INTEL")

    ax2.legend(
        bbox_to_anchor=(0, -0.75, 1, 0.2),
        loc="upper left",
        mode="expand",
        borderaxespad=0,
        ncol=3,
    )

    ax2.plot(train_t, m_intel - train_y, color=INTEL_COLOR, linestyle="--", alpha=0.4)
    ax2.plot(train_t, m_lintel - train_y, color=LINTEL_COLOR, alpha=0.4)

    ax2.set_ylabel("$m_n - y_n$")
    ax1.set_ylabel("y(t)")

    ax1.set_xlim([0, 3000])
    ax1.set_ylim([-5, 8])
    plt.suptitle("Synthetic Data, Outliers Only")
    plt.savefig(
        "plots/experiment_1_output.png",
        dpi=600,
        transparent=False,
        bbox_inches="tight",
    )

    plt.clf()
    plt.plot(w_intel[:, 0], label="0")
    plt.plot(w_intel[:, 1], label="1")
    plt.savefig("plots/experiment_1_intel_ws.png")
    plt.clf()
    plt.plot(w_lintel)
    plt.savefig("plots/experiment_1_lintel_ws.png")
