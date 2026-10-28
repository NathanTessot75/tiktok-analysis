"""Reproduces the final models of tiktok_likes_prediction.ipynb with the
hyperparameters selected by cross-validation, then draws the report figures.

Run from this folder:  python reproduce_models.py
Outputs: ../figures/*.pdf and results.json
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
FIG = os.path.join("..", "figures")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- data
df = pd.read_csv(os.path.join("..", "..", "data", "tiktok_dataset.csv"))
n_raw = len(df)
df = df.drop(columns=["#", "video_id"]).dropna()
n_clean = len(df)
likes_raw = df["video_like_count"].copy()

# Feature engineering (identical to the notebook)
df["share_per_view"] = df["video_share_count"] / (df["video_view_count"] + 1)
df["comments_per_view"] = df["video_comment_count"] / (df["video_view_count"] + 1)
df["downloads_per_view"] = df["video_download_count"] / (df["video_view_count"] + 1)
df["verified_status_encoded"] = df["verified_status"].map({"not verified": 0, "verified": 1}).fillna(-1)
df["author_ban_encoded"] = df["author_ban_status"].map({"active": 2, "under review": 1, "banned": 0}).fillna(-1)
df["claim_status_encoded"] = df["claim_status"].map({"not_claim": 0, "claim": 1}).fillna(-1)
df["text_len"] = df["video_transcription_text"].str.len()
df["transcript_word_count"] = df["video_transcription_text"].str.split().str.len()
df["transcript_avg_word_len"] = df["video_transcription_text"].apply(
    lambda x: np.mean([len(w) for w in str(x).split()]) if pd.notnull(x) and len(x.split()) > 0 else 0)
df["views_to_comment_ratio"] = df["video_view_count"] / (df["video_comment_count"] + 1)
df["views_to_download_ratio"] = df["video_view_count"] / (df["video_download_count"] + 1)
df["comment_to_download_ratio"] = df["video_comment_count"] / (df["video_download_count"] + 1)
df["share_to_download_ratio"] = df["video_share_count"] / (df["video_download_count"] + 1)
df["views_per_sec"] = df["video_view_count"] / (df["video_duration_sec"] + 1)
df["comments_per_sec"] = df["video_comment_count"] / (df["video_duration_sec"] + 1)
df["shares_per_sec"] = df["video_share_count"] / (df["video_duration_sec"] + 1)
df["log_views"] = np.log1p(df["video_view_count"])
df["log_comments"] = np.log1p(df["video_comment_count"])
df["log_shares"] = np.log1p(df["video_share_count"])
df["log_downloads"] = np.log1p(df["video_download_count"])
df["engagement_volume"] = (df["video_view_count"] + df["video_comment_count"]
                           + df["video_share_count"] + df["video_download_count"])
df["interactions_per_second"] = df["engagement_volume"] / (df["video_duration_sec"] + 1)
df = df.drop(columns=["video_transcription_text", "claim_status", "verified_status", "author_ban_status"])

X_cols = [c for c in df.columns if c != "video_like_count"]
df[X_cols] = StandardScaler().fit_transform(df[X_cols])
X, y = df[X_cols], df["video_like_count"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=666)

# ---------------------------------------------------------------- models
models = {
    "Linear regression": LinearRegression(),
    "LASSO regression": Lasso(alpha=1000, max_iter=10000),
    "Neural network": MLPRegressor(hidden_layer_sizes=(8, 4), activation="relu", solver="adam",
                                   learning_rate_init=0.01, batch_size=32, early_stopping=True,
                                   max_iter=500, random_state=666),
}
results, preds = {}, {}
for name, m in models.items():
    m.fit(X_train, y_train)
    p_tr, p_te = m.predict(X_train), m.predict(X_test)
    preds[name] = p_te
    results[name] = dict(train_rmse=float(np.sqrt(mean_squared_error(y_train, p_tr))),
                         test_rmse=float(np.sqrt(mean_squared_error(y_test, p_te))),
                         test_r2=float(r2_score(y_test, p_te)))

# Learning curve of the selected network, one epoch at a time (as in the notebook)
mlp = MLPRegressor(hidden_layer_sizes=(8, 4), activation="relu", solver="adam",
                   learning_rate_init=0.01, batch_size=32, early_stopping=True,
                   max_iter=1, warm_start=True, random_state=666)
train_curve, test_curve, loss_curve = [], [], []
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _ in range(500):
        mlp.fit(X_train, y_train)
        train_curve.append(float(np.sqrt(mean_squared_error(y_train, mlp.predict(X_train)))))
        loss_curve.append(float(np.sqrt(mlp.loss_)))
        test_curve.append(float(np.sqrt(mean_squared_error(y_test, mlp.predict(X_test)))))

summary = dict(n_raw=n_raw, n_clean=n_clean, n_features=len(X_cols),
               n_train=len(X_train), n_test=len(X_test),
               likes_mean=float(likes_raw.mean()), likes_median=float(likes_raw.median()),
               likes_sd=float(likes_raw.std()), test_sd=float(y_test.std()), results=results,
               curve_final=dict(train=train_curve[-1], test=test_curve[-1], sqrt_loss=loss_curve[-1]),
               curve_min_test=dict(iter=int(np.argmin(test_curve))+1, value=float(min(test_curve))))
json.dump(summary, open("results.json", "w"), indent=2)

# ---------------------------------------------------------------- style
for f in font_manager.findSystemFonts(["/usr/share/texmf/fonts/opentype/public/lm"]):
    font_manager.fontManager.addfont(f)
INK, INK2, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": ["Latin Modern Roman", "DejaVu Serif"], "font.size": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "pdf.fonttype": 42,
})
thousands = FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if abs(v) >= 1000 else f"{v:.0f}")

# Figure 1: distribution of likes
fig, ax = plt.subplots(figsize=(3.4, 2.4))
bins = np.logspace(0, np.log10(likes_raw.max() + 1), 35)
ax.hist(likes_raw + 1, bins=bins, color=BLUE, edgecolor="white", linewidth=0.6)
ax.set_xscale("log")
ax.set_xlabel("Likes (log scale)")
ax.set_ylabel("Videos")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_likes_distribution.pdf"), bbox_inches="tight")
plt.close(fig)

# Figure 2: predicted vs actual
fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.3), sharey=True)
lim = max(y_test.max(), max(p.max() for p in preds.values())) * 1.03
for ax, (name, p), letter in zip(axs, preds.items(), "abc"):
    ax.scatter(y_test, p, s=3, color=BLUE, alpha=0.18, linewidth=0, rasterized=True)
    ax.plot([0, lim], [0, lim], color=ORANGE, linewidth=1, linestyle="--")
    ax.set_xlim(-lim * 0.03, lim); ax.set_ylim(-lim * 0.08, lim)
    ax.xaxis.set_major_formatter(thousands); ax.yaxis.set_major_formatter(thousands)
    ax.set_title(f"({letter}) {name}", loc="left", fontsize=9, color=INK)
    ax.set_xlabel("Actual likes")
axs[0].set_ylabel("Predicted likes")
fig.tight_layout(w_pad=1.0)
fig.savefig(os.path.join(FIG, "fig2_predicted_vs_actual.pdf"), bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(FIG, "predicted_vs_actual.png"), bbox_inches="tight", dpi=200,
            facecolor="white")
plt.close(fig)

# Figure 3: learning curve
fig, ax = plt.subplots(figsize=(3.6, 2.4))
it = np.arange(1, len(train_curve) + 1)
ax.plot(it, train_curve, color=BLUE, linewidth=1.4, label="Train")
ax.plot(it, test_curve, color=ORANGE, linewidth=1.4, linestyle="--", label="Test")
ax.set_xlabel("Training iteration (epoch)")
ax.set_ylabel("RMSE (likes)")
ax.yaxis.set_major_formatter(thousands)
ax.set_ylim(40000, 60000)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_learning_curve.pdf"), bbox_inches="tight")
plt.close(fig)

print(json.dumps(summary, indent=2))
