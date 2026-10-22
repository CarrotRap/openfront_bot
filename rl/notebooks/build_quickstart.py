"""Generates notebooks/quickstart.ipynb.

Keeping the notebook's source as a plain script makes it reviewable and
diffable; run this file to regenerate the .ipynb after editing.
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# --------------------------------------------------------------------------

md(
    """
# OpenFront × PPO — prise en main

Environnement Gymnasium branché sur la **vraie** simulation d'OpenFront
(`OpenFrontIO/src/core`), jouée en headless dans un process Node.
L'agent affronte les bots/nations scriptés du jeu.

Au menu :
1. créer l'env et inspecter les espaces
2. visualiser l'observation spatiale
3. baseline aléatoire (masquée)
4. entraîner un `MaskablePPO`
5. évaluer

> Prérequis : `npm run inst` dans `OpenFrontIO/`, puis
> `pip install -r rl/requirements.txt`.
"""
)

code(
    """
import sys, time, collections
from pathlib import Path

RL_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RL_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from openfront_gym import OpenFrontEnv, sample_masked_action
print("rl root:", RL_ROOT)
"""
)

md(
    """
## 1. Créer l'environnement

Toute la config du jeu passe par un dict. Les cartes viennent soit des
fixtures de test (`mapSource="testdata"` : `plains` 100×100, `big_plains`
200×200, `world` 2000×1000) soit des cartes de production
(`mapSource="resources"`, ex. `oceania`, `onion`, `world`).

`big_plains` est le bon choix pour démarrer : petite, 100 % terrestre, rapide.
"""
)

code(
    """
CONFIG = {
    "map": "big_plains",      # fixture 200x200, tout en terre
    "mapSource": "testdata",
    "bots": 6,                # tribus scriptées adverses
    "nations": "disabled",    # pas de nations (la carte de test n'en définit pas)
    "difficulty": "Medium",
    "grid": 16,               # tête spatiale 16x16 = 256 cases
    "windowTiles": "adaptive",  # la fenêtre se dézoome quand le bot grandit
    "windowMinTiles": 64,     # zoom le plus serré
    "windowZoom": 2.5,        # côté de fenêtre = 2.5x l'étendue du territoire
    "maxPlayers": 8,          # K slots d'adversaires
    "ticksPerStep": 10,       # 1 décision = 1 seconde de jeu
    "maxSteps": 400,
}

env = OpenFrontEnv(config=CONFIG)
print("action heads :", env.spec_info["action"]["heads"])
print("nvec         :", env.nvec)
print("action types :", env.action_types)
print("ratios       :", env.ratios)
print("build units  :", env.build_units)
print()
print("observation  :")
for key, space in env.observation_space.spaces.items():
    print(f"  {key:18s} {space}")
"""
)

code(
    """
print(f"{len(env.channels)} canaux spatiaux :")
for i, c in enumerate(env.channels):
    print(f"  {i:2d}  {c}")
print()
print(f"{len(env.scalar_names)} scalaires globaux :")
print("  " + ", ".join(env.scalar_names))
print()
print(f"{len(env.player_features)} features par adversaire :")
print("  " + ", ".join(env.player_features))
"""
)

md(
    """
## 2. Reset et visualisation

Le reset joue la phase de spawn automatiquement : la politique ne voit que la
partie une fois tout le monde placé.
"""
)

code(
    """
obs, info = env.reset(seed=0)
print("info:", info)
print({k: v.shape for k, v in obs.items()})
"""
)

code(
    """
def plot_spatial(obs, env, cols=5):
    chans = env.channels
    rows = -(-len(chans) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.4 * rows))
    for ax, (i, name) in zip(axes.flat, enumerate(chans)):
        ax.imshow(obs["spatial"][i], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{i}. {name}", fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(chans):]:
        ax.axis("off")
    fig.tight_layout()
    return fig

plot_spatial(obs, env);
"""
)

md(
    """
## 3. Baseline : actions aléatoires masquées

`env.action_masks()` renvoie le masque aplati attendu par MaskablePPO.
`sample_masked_action` tire simplement une action valide au hasard — c'est la
baseline à battre.
"""
)

code(
    """
def rollout(env, policy, n_steps=400, seed=0, log_every=50):
    obs, _ = env.reset(seed=seed)
    kinds, total, invalid = collections.Counter(), 0.0, 0
    t0 = time.time()
    for i in range(n_steps):
        a = policy(obs)
        kinds[env.action_types[int(a[0])]] += 1
        obs, r, terminated, truncated, info = env.step(a)
        total += r
        invalid += bool(info.get("invalid"))
        if log_every and i % log_every == 0:
            print(f"  t={i:4d} r={r:+7.3f} | {env.render()}")
        if terminated or truncated:
            break
    dt = time.time() - t0
    print(f"  -> {i+1} steps en {dt:.1f}s ({(i+1)/dt:.0f} steps/s), "
          f"retour={total:.2f}, invalides={invalid/(i+1):.1%}")
    print(f"  -> actions: {dict(kinds)}")
    return total, i + 1, info

rng = np.random.default_rng(0)
rollout(env, lambda o: sample_masked_action(env, rng), n_steps=400, seed=0);
"""
)

code(
    """
returns = []
for seed in range(5):
    r, n, info = rollout(env, lambda o: sample_masked_action(env, rng),
                         n_steps=400, seed=seed, log_every=0)
    returns.append(r)
print(f"\\nbaseline aléatoire : {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
"""
)

md(
    """
## 4. Entraînement

`MaskablePPO` (sb3-contrib) gère nativement un `MultiDiscrete` masqué.
Le réseau `MultiBranchExtractor` traite séparément la carte (CNN), les
scalaires (MLP) et les adversaires (MLP partagé + pooling).

> 20 000 pas, c'est une démo. Pour un vrai modèle, sortez du notebook :
> `python train.py` ne bloque pas le kernel, monte à 8+ envs, écrit le sidecar
> de config à côté du checkpoint et sait repartir d'un modèle existant
> (`--init-from`). Voir la section « Entraîner une nouvelle version » du README.
"""
)

code(
    """
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecMonitor
from openfront_gym import OpenFrontVecEnv
from openfront_gym.policy import DEFAULT_POLICY_KWARGS

# OpenFrontVecEnv, pas DummyVecEnv : chaque step() bloque sur un aller-retour
# vers son process Node, et DummyVecEnv les appelle en boucle — N envs
# tourneraient à la vitesse d'un seul dès que la carte grossit.
N_ENVS = 4
make = lambda i: (lambda: OpenFrontEnv(config=CONFIG, seed_prefix=f"w{i}"))
venv = VecMonitor(OpenFrontVecEnv([make(i) for i in range(N_ENVS)]))

model = MaskablePPO(
    "MultiInputPolicy", venv,
    policy_kwargs=DEFAULT_POLICY_KWARGS,
    n_steps=256, batch_size=512, ent_coef=0.01,
    learning_rate=3e-4, verbose=1, device="cpu",
)
print(f"{sum(p.numel() for p in model.policy.parameters()):,} paramètres")
"""
)

code(
    """
model.learn(total_timesteps=20_000, progress_bar=True)
model.save("ppo_openfront_demo")
"""
)

md(
    """
## 5. Évaluation

On compare le retour moyen de la politique apprise à la baseline aléatoire,
sur les mêmes graines.
"""
)

code(
    """
def policy_action(obs):
    action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
    return action

trained = []
for seed in range(5):
    r, n, info = rollout(env, policy_action, n_steps=400, seed=seed, log_every=0)
    trained.append(r)

print(f"\\naléatoire : {np.mean(returns):7.2f} +/- {np.std(returns):.2f}")
print(f"PPO       : {np.mean(trained):7.2f} +/- {np.std(trained):.2f}")
"""
)

code(
    """
obs, _ = env.reset(seed=42)
for _ in range(200):
    obs, r, te, tr, info = env.step(policy_action(obs))
    if te or tr:
        break
print(env.render())
plot_spatial(obs, env);
"""
)

md(
    """
## 6. Et après

Leviers, du plus rentable au moins rentable :

| Levier | Où |
|---|---|
| Poids de récompense (`rewardWeights`) | `CONFIG["rewardWeights"]` |
| Fréquence de décision (`ticksPerStep`) | 10 = 1 s de jeu ; 5 pour plus de réactivité |
| Taille de la tête spatiale (`grid`) | 16 → 256 actions ; 32 → 1024 (plus dur) |
| Difficulté / nombre de bots | curriculum : `bots` 2 → 6 → 12 |
| Carte | `big_plains` → `world` (eau, donc débarquements) |
| Nouvelles actions | ajouter un cas dans `bridge/src/actions.ts` + `spec.ts` |

Pour ajouter une action (alliance, donation, nuke…) : déclarez-la dans
`ACTION_TYPES` (`bridge/src/spec.ts`), construisez l'intent correspondant dans
`ActionCodec.decode` et son masque dans `ActionCodec.masks`. La liste complète
des intents du jeu est dans `OpenFrontIO/src/core/Schemas.ts`.
"""
)

code(
    """
venv.close()
env.close()
"""
)

# --------------------------------------------------------------------------

notebook = {
    "cells": [
        {
            "cell_type": kind,
            "id": f"cell-{i:02d}",
            "metadata": {},
            "source": body.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if kind == "code" else {}),
        }
        for i, (kind, body) in enumerate(CELLS)
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).with_name("quickstart.ipynb")
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {out} ({len(CELLS)} cells)")
