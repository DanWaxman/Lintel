import jax

jax.config.update("jax_enable_x64", True)

from Lintel import lintel, intel
from Lintel.gp_utils import GP, MarkovianGP
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="paper", style="whitegrid", palette="colorblind")

s = "data/noumenta/realAWSCloudwatch/realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv"
train_y = pd.read_csv(s).value.to_numpy()
train_t = np.arange(len(train_y))

pretrain_t = train_t[:100]
train_t = train_t[100:]

pretrain_y = train_y[:100]
train_y = train_y[100:]

FIT_MODE = True
PLOT_MODE = True

if FIT_MODE:
    gp1 = GP(
        np.array(pretrain_t),
        np.array(pretrain_y),
        lengthscale=10.0,
        sigma_f=1.0,
        sigma_n=pretrain_y.std(),
        C=pretrain_y.mean(),
    )

    print("Pretraining GP")
    gp1.maximize_evidence(pretrain_t, pretrain_y, lr=1e-2, iters=4500, verbose=False)
    print("Finished Pretraining")

    gp1 = MarkovianGP(
        lengthscale=gp1.lengthscale,
        sigma_f=gp1.sigma_f,
        sigma_n=gp1.sigma_n,
        C=pretrain_y.mean(),
    )
    gp1.reset_and_filter(pretrain_t, pretrain_y, pretrain_y.mean())

    gp2 = MarkovianGP(
        lengthscale=gp1.lengthscale,
        sigma_f=gp1.sigma_f / 5.0,
        sigma_n=gp1.sigma_n,
        C=pretrain_y.mean(),
    )
    gp2.reset_and_filter(pretrain_t, pretrain_y, pretrain_y.mean())

    gp3 = MarkovianGP(
        lengthscale=gp1.lengthscale,
        sigma_f=gp1.sigma_f,
        sigma_n=gp1.sigma_n / 5.0,
        C=pretrain_y.mean(),
    )
    gp3.reset_and_filter(pretrain_t, pretrain_y, pretrain_y.mean())

    gp4 = MarkovianGP(
        lengthscale=gp1.lengthscale,
        sigma_f=gp1.sigma_f / 2.0,
        sigma_n=gp1.sigma_n / 2.0,
        C=pretrain_y.mean(),
    )
    gp4.reset_and_filter(pretrain_t, pretrain_y, pretrain_y.mean())

    lintel = lintel.LINTEL(
        N=3,
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
        m, s, o = lintel.predict_and_update(train_t[t], train_y[t])
        ws.append(lintel.weights)

        if o:
            ots.append(t)

        ms.append(m)
        ss.append(s)

    ms = np.array(ms)
    ss = np.array(ss)
    ots = np.array(ots)

    np.savez(
        "results/ec2_cpu_utilization_ac20cd_lintel_results.npz", ms=ms, ss=ss, ots=ots
    )

    gp1 = GP(
        np.array(pretrain_t),
        np.array(pretrain_y),
        lengthscale=10.0,
        sigma_f=1.0,
        sigma_n=pretrain_y.std(),
        C=pretrain_y.mean(),
    )

    gp1.maximize_evidence(pretrain_t, pretrain_y, lr=1e-2, iters=4500, verbose=False)

    gp2 = GP(
        np.array(pretrain_t),
        np.array(pretrain_y),
        lengthscale=gp1.lengthscale,
        sigma_f=gp1.sigma_f / 2.0,
        sigma_n=gp1.sigma_n,
        C=pretrain_y.mean(),
    )

    gp3 = GP(
        np.array(pretrain_t),
        np.array(pretrain_y),
        lengthscale=gp1.lengthscale,
        sigma_f=gp1.sigma_f,
        sigma_n=gp1.sigma_n / 2.0,
        C=pretrain_y.mean(),
    )

    gp4 = GP(
        np.array(pretrain_t),
        np.array(pretrain_y),
        lengthscale=gp1.lengthscale,
        sigma_f=gp1.sigma_f / 2.0,
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
    ots = np.array(ots)

    np.savez(
        "results/ec2_cpu_utilization_ac20cd_intel_results.npz", ms=ms, ss=ss, ots=ots
    )

### Plotting
if PLOT_MODE:
    res_lintel = np.load("results/ec2_cpu_utilization_ac20cd_lintel_results.npz")
    m_lintel = res_lintel["ms"]
    s_lintel = res_lintel["ss"]
    o_lintel = res_lintel["ots"]

    res_intel = np.load("results/ec2_cpu_utilization_ac20cd_intel_results.npz")
    m_intel = res_intel["ms"]
    s_intel = res_intel["ss"]
    o_intel = res_intel["ots"]

    fig, ax1 = plt.subplots(figsize=(7, 2))
    ax1.scatter(pretrain_t, pretrain_y, alpha=0.2, label="Pretraining Points")
    ax1.scatter(train_t, train_y, color="black", alpha=0.2, label="Training Points")
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
    ax1.plot(train_t, m_lintel, label="INTEL")
    ax1.fill_between(
        train_t,
        m_lintel + 2 * np.sqrt(s_lintel),
        m_lintel - 2 * np.sqrt(s_lintel),
        alpha=0.3,
    )
    ax1.plot(train_t, m_intel, linestyle="--", label="LINTEL")
    ax1.fill_between(
        train_t,
        m_intel + 2 * np.sqrt(s_intel),
        m_intel - 2 * np.sqrt(s_intel),
        alpha=0.3,
    )

    x1, x2, y1, y2 = (
        train_t[450],
        train_t[650],
        np.min(m_lintel[450:650] - 2 * np.sqrt(s_lintel)[450:650]) - 1.0,
        np.max(m_lintel[450:650] + 2 * np.sqrt(s_lintel)[450:650]) + 1.0,
    )
    axin1 = ax1.inset_axes(
        [0.2, 0.65, 0.3, 0.3],
        xlim=(x1, x2),
        ylim=(y1, y2),
        xticks=[],
        yticks=[],
        xticklabels=[],
        yticklabels=[],
    )

    axin1.scatter(pretrain_t[450:650], pretrain_y[450:650], alpha=0.2)
    axin1.scatter(train_t[450:650], train_y[450:650], color="black", alpha=0.2)
    axin1.plot(train_t[450:650], m_lintel[450:650])
    axin1.fill_between(
        train_t[450:650],
        m_lintel[450:650] + 2 * np.sqrt(s_lintel)[450:650],
        m_lintel[450:650] - 2 * np.sqrt(s_lintel)[450:650],
        alpha=0.3,
    )
    axin1.plot(train_t[450:650], m_intel[450:650], linestyle="--")
    axin1.fill_between(
        train_t[450:650],
        m_intel[450:650] + 2 * np.sqrt(s_intel)[450:650],
        m_intel[450:650] - 2 * np.sqrt(s_intel)[450:650],
        alpha=0.3,
    )
    os = o_lintel[(o_lintel > 450) * (o_lintel < 650)]
    axin1.scatter(
        train_t[os],
        train_y[os],
        marker="x",
        color="red",
        alpha=0.8,
        label="LINTEL Outlier",
    )
    os = o_intel[(o_intel > 450) * (o_intel < 650)]
    axin1.scatter(
        train_t[os],
        train_y[os],
        marker="+",
        color="red",
        alpha=0.8,
        label="INTEL Outlier",
    )

    ax1.indicate_inset_zoom(axin1, edgecolor="black", alpha=1, lw=0.7)

    plt.title("EC2 Dataset")
    plt.savefig(
        "plots/experiment_2_output.png",
        dpi=600,
        transparent=False,
        bbox_inches="tight",
    )
