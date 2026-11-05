"""
Projet 1 : analyse risque / rendement d'actions françaises
Installation : pip install yfinance pandas numpy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# --- 1. Paramètres -----------------------------------------------------
TICKERS = {
    "MC.PA": "LVMH",
    "TTE.PA": "TotalEnergies",
    "BNP.PA": "BNP Paribas",
    "GLE.PA": "Société Générale",
    "AIR.PA": "Airbus",
    "^FCHI": "CAC 40",
}
START = "2020-01-01"
END = "2026-11-01"
TRADING_DAYS = 252  # nombre de jours de bourse par an

# --- 2. Données --------------------------------------------------------
prices = yf.download(list(TICKERS), start=START, end=END)["Close"]
prices = prices.rename(columns=TICKERS).dropna()

# Rendements journaliers : (P_t / P_{t-1}) - 1
returns = prices.pct_change().dropna()

# --- 3. Statistiques ---------------------------------------------------
stats = pd.DataFrame({
    "Rendement annualisé": returns.mean() * TRADING_DAYS,
    "Volatilité annualisée": returns.std() * np.sqrt(TRADING_DAYS),
})
stats["Rendement / risque"] = (
    stats["Rendement annualisé"] / stats["Volatilité annualisée"]
)
print(stats.round(3))

corr = returns.corr()
print("\nMatrice de corrélation :")
print(corr.round(2))

# --- 4. Graphiques -----------------------------------------------------
# a) Performance en base 100
base100 = prices / prices.iloc[0] * 100
base100.plot(figsize=(10, 5), title="Performance (base 100)")
plt.ylabel("Valeur")
plt.tight_layout()
plt.savefig("performance_base100.png", dpi=150)
plt.close()

# b) Matrice de corrélation
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
fig.colorbar(im)
ax.set_title("Corrélation des rendements journaliers")
plt.tight_layout()
plt.savefig("correlations.png", dpi=150)
plt.close()
