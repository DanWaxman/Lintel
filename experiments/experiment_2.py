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

if FIT_MODE:
    np.random.seed(SEED)

    print("Fetching Data")
    s = "data/noumenta/realAWSCloudwatch/realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv"
    y = pd.read_csv(s).value.to_numpy()
    t = np.arange(len(y))

    pretrain_t = t[:250].reshape(-1, 1)
    train_t = t[250:].reshape(-1, 1)

    pretrain_y = y[:250]
    train_y = y[250:]
    print("Finished Generating Data")

    np.savez(
        f"results/experiment_2_seed_{SEED}_geom_{args.geometric_fusion}_data.npz",
        pretrain_t=pretrain_t,
        train_t=train_t,
        pretrain_y=pretrain_y,
        train_y=train_y,
    )

    k0 = bayesnewton.kernels.Matern32()

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
    k2 = bayesnewton.kernels.Matern32(
        lengthscale=k0.lengthscale, variance=k0.variance * 2
    )
    k3 = bayesnewton.kernels.Matern32(lengthscale=k0.lengthscale, variance=k0.variance)
    k4 = bayesnewton.kernels.Matern32(
        lengthscale=k0.lengthscale, variance=k0.variance * 2
    )
    k5 = bayesnewton.kernels.Matern32(
        lengthscale=k0.lengthscale, variance=k0.variance / 2.0
    )
    k6 = bayesnewton.kernels.Matern32(
        lengthscale=k0.lengthscale, variance=k0.variance / 2.0
    )
    k7 = bayesnewton.kernels.Matern32(
        lengthscale=k0.lengthscale, variance=k0.variance * 2
    )
    k8 = bayesnewton.kernels.Matern32(
        lengthscale=k0.lengthscale, variance=k0.variance / 2.0
    )

    gp1 = MarkovianGP(C=pretrain_y.mean(), sigma_n=gp0.sigma_n, kernel=k1)
    gp1.reset_and_filter(pretrain_t.squeeze()[-1:], pretrain_y[-1:], pretrain_y.mean())

    gp2 = MarkovianGP(C=pretrain_y.mean(), sigma_n=gp0.sigma_n, kernel=k2)
    gp2.reset_and_filter(pretrain_t.squeeze()[-1:], pretrain_y[-1:], pretrain_y.mean())

    gp3 = MarkovianGP(C=pretrain_y.mean(), sigma_n=gp0.sigma_n * np.sqrt(2), kernel=k3)
    gp3.reset_and_filter(pretrain_t.squeeze()[-1:], pretrain_y[-1:], pretrain_y.mean())

    gp4 = MarkovianGP(C=pretrain_y.mean(), sigma_n=gp0.sigma_n * np.sqrt(2), kernel=k4)
    gp4.reset_and_filter(pretrain_t.squeeze()[-1:], pretrain_y[-1:], pretrain_y.mean())

    gp5 = MarkovianGP(C=pretrain_y.mean(), sigma_n=gp0.sigma_n, kernel=k5)
    gp5.reset_and_filter(pretrain_t.squeeze()[-1:], pretrain_y[-1:], pretrain_y.mean())

    gp6 = MarkovianGP(C=pretrain_y.mean(), sigma_n=gp0.sigma_n / np.sqrt(2), kernel=k6)
    gp6.reset_and_filter(pretrain_t.squeeze()[-1:], pretrain_y[-1:], pretrain_y.mean())

    gp7 = MarkovianGP(C=pretrain_y.mean(), sigma_n=gp0.sigma_n / np.sqrt(2), kernel=k7)
    gp7.reset_and_filter(pretrain_t.squeeze()[-1:], pretrain_y[-1:], pretrain_y.mean())

    gp8 = MarkovianGP(C=pretrain_y.mean(), sigma_n=gp0.sigma_n * np.sqrt(2), kernel=k8)
    gp8.reset_and_filter(pretrain_t.squeeze()[-1:], pretrain_y[-1:], pretrain_y.mean())

    print("##################")
    print("#### LINTEL #####")
    print("##################")
    lintel = lintel.LINTEL(
        N=3,
        alpha=0.9,
        weights=np.ones(
            8,
        )
        / 8,
        gps=[gp1, gp2, gp3, gp4, gp5, gp6, gp7, gp8],
        L=50,
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
        f"results/experiment_2_seed_{SEED}_geom_{args.geometric_fusion}_lintel_results.npz",
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
        pretrain_t, pretrain_y, C=pretrain_y.mean(), sigma_n=gp0.sigma_n, kernel=k2
    )
    gp3 = GP(
        pretrain_t,
        pretrain_y,
        C=pretrain_y.mean(),
        sigma_n=gp0.sigma_n * np.sqrt(2),
        kernel=k3,
    )
    gp4 = GP(
        pretrain_t,
        pretrain_y,
        C=pretrain_y.mean(),
        sigma_n=gp0.sigma_n * np.sqrt(2),
        kernel=k4,
    )
    gp5 = GP(
        pretrain_t, pretrain_y, C=pretrain_y.mean(), sigma_n=gp0.sigma_n, kernel=k5
    )
    gp6 = GP(
        pretrain_t,
        pretrain_y,
        C=pretrain_y.mean(),
        sigma_n=gp0.sigma_n / np.sqrt(2),
        kernel=k6,
    )
    gp7 = GP(
        pretrain_t,
        pretrain_y,
        C=pretrain_y.mean(),
        sigma_n=gp0.sigma_n / np.sqrt(2),
        kernel=k7,
    )
    gp8 = GP(
        pretrain_t,
        pretrain_y,
        C=pretrain_y.mean(),
        sigma_n=gp0.sigma_n * np.sqrt(2),
        kernel=k8,
    )

    print("#################")
    print("#### INTEL #####")
    print("#################")
    intel = intel.INTEL(
        N=3,
        tau=20,
        alpha=0.9,
        weights=np.ones(
            8,
        )
        / 8,
        gps=[gp1, gp2, gp3, gp4, gp5, gp6, gp7, gp8],
        L=50,
        verbose=True,
        product_of_experts=True,
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
        f"results/experiment_2_seed_{SEED}_geom_{args.geometric_fusion}_intel_results.npz",
        ms=ms,
        ss=ss,
        ots=ots,
        ws=ws,
        t_intel=t_intel,
    )

### Plotting
if PLOT_MODE:
    data = np.load(
        f"results/experiment_2_seed_{SEED}_geom_{args.geometric_fusion}_data.npz"
    )
    train_t = data["train_t"].squeeze()
    pretrain_t = data["pretrain_t"].squeeze()
    train_y = data["train_y"]
    pretrain_y = data["pretrain_y"]

    res_lintel = np.load(
        f"results/experiment_2_seed_{SEED}_geom_{args.geometric_fusion}_lintel_results.npz"
    )
    m_lintel = res_lintel["ms"]
    s_lintel = res_lintel["ss"]
    o_lintel = res_lintel["ots"]
    w_lintel = res_lintel["ws"]

    res_intel = np.load(
        f"results/experiment_2_seed_{SEED}_geom_{args.geometric_fusion}_intel_results.npz"
    )
    m_intel = res_intel["ms"]
    s_intel = res_intel["ss"]
    o_intel = res_intel["ots"]
    w_intel = res_intel["ws"]

    def moving_average(x, w=10):
        return np.convolve(x, np.ones(w), "valid") / w

    intel_pll = stats.norm.logpdf(train_y, m_intel, np.sqrt(s_intel))
    lintel_pll = stats.norm.logpdf(train_y, m_lintel, np.sqrt(s_lintel))
    print(intel_pll[750:1250].mean())
    print(lintel_pll[750:1250].mean())

    intel_nmse = (train_y - m_intel) ** 2 / np.var(train_y)
    lintel_nmse = (train_y - m_lintel) ** 2 / np.var(train_y)

    plt.scatter(range(len(moving_average(intel_pll))), moving_average(intel_pll))
    plt.scatter(range(len(moving_average(lintel_pll))), moving_average(lintel_pll))
    plt.savefig("plots/experiment_2_plls.png")
    plt.clf()
    print(train_t.shape, intel_pll.shape)
    plt.scatter(train_t, intel_pll - lintel_pll)
    plt.savefig("plots/experiment_2_pll_diffs.png")

    fig, axs = plt.subplots(
        nrows=2, figsize=(7.5, 2), height_ratios=[2, 1], sharex=True
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

    zoom_start = 422 - 250 - 100
    zoom_end = 422 - 250 + 400
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
        [0.4, 0.65, 0.15, 0.3],
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

    # plt.legend(
    #     bbox_to_anchor=(0, -0.4, 1, 0.2),
    #     loc="upper left",
    #     mode="expand",
    #     borderaxespad=0,
    #     ncol=3,
    # )

    ax2.plot(train_t, m_intel - train_y, color=INTEL_COLOR, linestyle="--", alpha=0.4)
    ax2.plot(train_t, m_lintel - train_y, color=LINTEL_COLOR, alpha=0.4)

    ax1.set_xlim([0, 3782])
    plt.suptitle("CPU Utilization")
    ax2.set_ylabel("$m_n - y_n$")
    ax1.set_ylabel("y(t)")

    plt.savefig(
        "plots/experiment_2_output.png",
        dpi=600,
        transparent=False,
        bbox_inches="tight",
    )

    plt.clf()
    plt.plot(w_intel[:, 0], label="0")
    plt.plot(w_intel[:, 1], label="1")
    plt.legend()
    plt.savefig("plots/experiment_2_intel_ws.png")
    plt.clf()
    plt.plot(w_lintel)
    plt.savefig("plots/experiment_2_lintel_ws.png")
