# OpenFront RL — environnement Gymnasium + PPO

Un agent PPO qui joue à [OpenFront.io](../OpenFrontIO) contre les bots du jeu.
L'environnement pilote la **vraie** simulation (`OpenFrontIO/src/core`) en
headless, sans client ni serveur : mêmes règles, même économie, même combat,
même IA adverse.

```
openfront_bot/
├── OpenFrontIO/            le jeu (non modifié)
└── rl/
    ├── bridge/             pont TypeScript — pilote la simulation headless
    │   └── src/
    │       ├── spec.ts         layout observation/action (source de vérité)
    │       ├── observation.ts  encodeur des tenseurs
    │       ├── render.ts        image de la carte pour le visualiseur
    │       ├── actions.ts      masques + décodage action → Intent
    │       ├── env.ts          boucle de partie, récompense, terminaison
    │       ├── protocol.ts     framing binaire stdio
    │       └── main.ts         point d'entrée (un process = un env)
    ├── openfront_gym/      paquet Python
    │   ├── bridge.py           transport vers le process Node
    │   ├── env.py              gym.Env
    │   ├── vec.py              VecEnv pipeliné (simulations concurrentes)
    │   └── policy.py           extracteur de features multi-branches
    ├── tests/               test d'équivalence du VecEnv
    ├── viewer/             visualiseur web temps réel
    │   ├── server.py           serveur SSE (stdlib uniquement)
    │   └── index.html          page (canvas, sans build ni CDN)
    ├── notebooks/quickstart.ipynb
    ├── train.py            entraînement CLI
    └── eval.py             comparaison politique vs baseline, mêmes graines
```

## Installation

[OpenFrontIO](https://github.com/openfrontio/OpenFrontIO) est épinglé en
submodule sur la commit `af68a46de`. OpenFrontIO évolue vite et son
`src/core/` a été restructuré après cette commit (split en packages
engine/engine-api/shared), ce qui casse les imports du bridge TypeScript.

Install complet depuis zéro :

```bash
git clone --recursive https://github.com/CarrotRap/openfront_bot.git
cd openfront_bot/OpenFrontIO && npm install && cd ../rl
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/jupyter notebook notebooks/quickstart.ipynb
```

Si tu as cloné sans `--recursive`, rattrape le submodule avant le reste :

```bash
git submodule update --init
```

Vérification rapide :

```python
from openfront_gym import OpenFrontEnv, sample_masked_action
import numpy as np

env = OpenFrontEnv(config={"map": "big_plains", "bots": 6, "maxSteps": 400})
obs, info = env.reset(seed=0)
rng = np.random.default_rng(0)
for _ in range(400):
    obs, r, terminated, truncated, info = env.step(sample_masked_action(env, rng))
    if terminated or truncated:
        break
print(env.render())
env.close()
```

## Observation

`gym.spaces.Dict` à trois branches, plus les masques.

| Clé | Shape | Contenu |
|---|---|---|
| `spatial` | `(15, G, G)` | vue égocentrée de la carte, 15 canaux |
| `scalar` | `(46,)` | état global de l'agent |
| `players` | `(K, 16)` | les K adversaires les plus pertinents |
| `mask_*` | — | masques par tête d'action |

**Canaux spatiaux** — `land`, `ocean`, `shoreline`, `owner_self`, `owner_ally`,
`owner_enemy`, `owner_neutral_land`, `self_border`, `owner_strength`,
`self_structure`, `enemy_structure`, `fallout`, `buildable`, `boat_target`,
`off_map`.

La vue est une fenêtre carrée centrée sur le plus gros bloc de territoire de
l'agent, rééchantillonnée en `G × G`.

**La fenêtre se dézoome avec l'agent** (`windowTiles: "adaptive"`, le défaut).
Son côté suit l'étendue linéaire du territoire — le plus grand de la boîte
englobante et de √(surface), pour couvrir aussi bien un empire compact qu'un
empire allongé — multiplié par `windowZoom`.

Les tailles sont des **octaves discrètes** (64, 128, 256, 512 … jusqu'à la
carte) et non une échelle continue, avec hystérésis : on dézoome dès que
l'agent dépasse le niveau courant, on rezoome seulement quand il tient
confortablement (85 %) dans le niveau inférieur. Une échelle continue
remapperait le même terrain sur une case différente à chaque pas, et aucune
case ne pourrait acquérir de sens stable — ni pour le CNN, ni pour la tête
d'action positionnelle.

Deux scalaires (`window_scale`, `window_frac`) disent à la politique à quel
niveau de zoom elle regarde : sans eux, « 8 cases vertes » serait ambigu entre
un petit empire vu de près et un grand vu de loin.

Le territoire de l'agent et les structures sont peints case par case en plus de
l'échantillonnage : une fois dézoomée, une cellule couvre des dizaines de
tuiles et un petit territoire disparaîtrait sinon complètement.

**Scalaires** — territoire, troupes / `maxTroops`, or, revenu, rang, attaques
entrantes et sortantes, alliances, doomsday clock, immunité de spawn, puis
pour chacune des 7 unités constructibles : possédées, abordables, plaçables.
Noms exacts dans `env.scalar_names`.

**Adversaires** — taille relative, frontière commune, relation, alliance,
équipe, ciblage réciproque, attaques en cours, traître, type, distance.
Le slot 0 de chaque vecteur est un flag de validité (les slots vides sont
ignorés par le pooling).

## Actions

`MultiDiscrete([4, K+1, 6, 7, G*G])` — cinq têtes indépendantes :

| Tête | Taille | Rôle |
|---|---|---|
| `action_type` | 4 | `NOOP`, `ATTACK`, `BOAT_ATTACK`, `BUILD` |
| `target_slot` | K+1 | 0 = TerraNullius, 1..K = adversaires |
| `ratio` | 6 | 5 %, 10 %, 20 %, 35 %, 50 %, 80 % des troupes |
| `unit_type` | 7 | City, Port, Factory, DefensePost, MissileSilo, SAMLauncher, Warship |
| `cell` | G² | case de la grille (build / débarquement) |

Les têtes non pertinentes pour l'`action_type` tiré sont ignorées. C'est une
approximation assumée : la politique reste une distribution produit, ce qui
évite une tête auto-régressive. Les combinaisons illégales qui passent malgré
les masques sont rejetées par le moteur et signalées via `info["invalid"]`,
avec une petite pénalité.

`ATTACK` n'a pas de position : dans OpenFront l'intent d'attaque ne porte
qu'une cible et un nombre de troupes (`Schemas.ts:617`), la poussée se
propageant sur toute la frontière commune.

**Masques.** `env.action_masks()` renvoie le masque aplati attendu par
`MaskablePPO`. `env.masks_by_head()` renvoie le détail, plus les masques
conditionnels `mask_cell_build` / `mask_cell_boat` si vous voulez une
politique auto-régressive.

## Récompense

Calculée dans `bridge/src/env.ts` (`step()`), à chaque décision :

```
r = w_tiles  · (tuiles_t − tuiles_{t−1}) / part_équitable · 10
  + w_troops · (log1p(troupes_t) − log1p(troupes_{t−1}))
  + w_gold   · (log1p(or_t)      − log1p(or_{t−1}))
  + w_step                          coût de temps, à chaque pas
  + w_invalid                       si le moteur a rejeté l'action
  + w_death                         terminal : l'agent est mort
  + w_win                           terminal : l'agent a gagné
```

| Poids | Défaut | Rôle |
|---|---|---|
| `tiles` | 1.0 | **le signal principal** |
| `troops` | 0.05 | garder une armée, ne pas tout dépenser |
| `gold` | 0.02 | accumuler de quoi construire |
| `step` | −0.0005 | préférer gagner tôt |
| `invalid` | −0.02 | décourager les combinaisons illégales |
| `death` | −1.0 | terminal |
| `win` | +5.0 | terminal |

### Pourquoi c'est construit comme ça

**Les termes de façonnage télescopent.** Ce sont des différences : leur somme
sur un épisode se réduit à `(état_final − état_initial)`, indépendamment du
chemin pris. Le terme territoire vaut donc exactement
`tuiles_finales / part_équitable · 10` sur tout l'épisode. La politique ne
peut pas grappiller de la récompense en faisant des allers-retours — seul
l'état final compte, ce qui est la propriété qu'on veut d'un shaping.

Mesuré sur `big_plains`, 7 joueurs (part équitable = 5 714 tuiles) :

| issue | pas | tuiles finales | retour | terme territoire seul | reste |
|---|---|---|---|---|---|
| tronqué | 400 | 4 383 | +7.32 | +7.67 | −0.35 |
| victoire | 160 | 32 026 | +60.97 | +56.05 | +4.93 |
| tronqué | 400 | 3 765 | +6.25 | +6.59 | −0.34 |

Le « reste » est le bonus de victoire, les termes troupes/or et le coût de
temps. Tout le reste du retour, c'est le territoire.

**Mourir revient à tout perdre, pas à être puni deux fois.** À la mort,
`tuiles_t = 0` : le terme territoire encaisse d'un coup la perte de tout ce
qui avait été accumulé, et la somme télescopée retombe à ~0. Le `w_death` de
−1 n'est donc qu'un complément symbolique, pas la vraie sanction — la vraie
sanction est de renoncer aux +56 qu'on avait en banque.

**Or et troupes sont en log**, pas en linéaire : rendements décroissants et
invariance d'échelle. Passer de 1 000 à 2 000 troupes vaut autant que de
10 000 à 20 000, ce qui évite que ces termes secondaires n'écrasent le
territoire en fin de partie.

**Le territoire est normalisé par la « part équitable »**
(`tuiles_terrestres / nombre_de_joueurs`, figée au début de l'épisode), pas par
la taille de la carte. C'est ce qui rend la récompense comparable d'une carte
à l'autre : posséder 5× sa part équitable rapporte +50 partout.

| carte | joueurs | tuiles (5× la part) | `map_share` | `fair_share` |
|---|---|---|---|---|
| `big_plains` | 7 | 28 571 | 71.4 | **50.0** |
| `world` | 41 | 79 460 | 12.2 | **50.0** |
| `world` | 103 | 31 630 | 4.9 | **50.0** |

Avec l'ancienne normalisation (`tilesRewardMode: "map_share"`, toujours
disponible en config), le même succès relatif valait 71 sur une petite carte et
4.9 sur une grande : le terme territoire y devenait 5× plus faible que la
pénalité de mort, et l'agent n'optimisait plus que « ne pas mourir ».

### Limites à connaître

- **Pic de variance à la mort** : la perte de territoire arrive en un seul pas
  (−56 d'un coup). GAE l'absorbe, mais ça fait de gros gradients ; baisser
  `w_tiles` ou passer à une récompense de territoire bornée aide si
  l'entraînement diverge.
- **`w_win` est rarement atteint** dans les configs par défaut : l'épisode est
  généralement tronqué avant. Il ne guide rien tant que `maxSteps` est court.
- **Aucun signal diplomatique** : alliances, embargos et donations ne sont pas
  dans l'espace d'action de la version starter, donc rien ne les récompense.
- **Au-delà de ~40 adversaires, il n'y a plus rien à apprendre** : la politique
  aléatoire meurt dans 100 % des épisodes, donc toutes les trajectoires ont le
  même retour et le gradient est nul. Voir « Passer à une grande carte ».

L'épisode se termine quand l'agent meurt ou qu'un vainqueur est désigné, et
est tronqué à `maxSteps`.

## Configuration

| Clé | Défaut | Effet |
|---|---|---|
| `map` / `mapSource` | `big_plains` / `testdata` | `testdata` : `plains`, `big_plains`, `world`. `resources` : toutes les cartes du jeu |
| `bots` | 6 | tribus scriptées adverses |
| `nations` | `disabled` | `default` / `disabled` / nombre |
| `difficulty` | `Medium` | agressivité et temps de réaction des nations |
| `grid` | 32 | côté de la grille — 16 donne une tête spatiale de 256, bien plus facile à apprendre |
| `windowTiles` | `adaptive` | `adaptive` = dézoom automatique ; `auto` = carte entière ; un entier = fenêtre fixe |
| `windowMinTiles` | 64 | zoom le plus serré (début de partie) |
| `windowMaxTiles` | 0 | 0 = plafonné au plus grand côté de la carte |
| `windowZoom` | 2.5 | côté de fenêtre / étendue du territoire |
| `maxPlayers` | 8 | K slots d'adversaires |
| `ticksPerStep` | 10 | 10 ticks = 1 s de jeu par décision |
| `maxSteps` | 1200 | troncature |
| `instantBuild`, `infiniteGold` | `false` | utiles pour déboguer l'action `BUILD` |
| `tilesRewardMode` | `fair_share` | normalisation du territoire ; `map_share` pour l'ancien comportement |
| `emitRender`, `renderMaxDim` | `false` / 256 | image de la carte pour le visualiseur web |

## Passer à une grande carte

Changer `CONFIG` suffit pour que **ça tourne** — les cartes de production se
chargent (`mapSource: "resources"`), les nations se créent depuis le manifest
(`nations: "default"`), et un process Node ne prend que **56 Mo** même sur
2000×1000. Mais trois choses demandent une intervention.

### 1. `DummyVecEnv` annule tout le parallélisme

Chaque `step()` bloque sur un aller-retour vers son process Node.
`DummyVecEnv` les appelle en boucle, donc N envs tournent à la vitesse d'un
seul. Sur `big_plains` c'est invisible (la simulation est ~1000× plus rapide
que le tuyau) ; sur une grande carte c'est tout le coût.

`OpenFrontVecEnv` (`openfront_gym/vec.py`) envoie les N commandes d'abord et
collecte les N réponses ensuite : les simulations se recouvrent. Ni process
Python supplémentaire ni pickling, contrairement à `SubprocVecEnv`.

Mesuré sur `world` (2000×1000, 30 bots + 72 nations), 8 envs :

| | pas-env/s | par pas vectorisé |
|---|---|---|
| `DummyVecEnv` | 86 | 93 ms |
| `OpenFrontVecEnv` | **300** | 27 ms |

C'est le défaut de `train.py` (`--vec pipelined | dummy | subproc`).

### 2. Monter `maxPlayers` et `maxSteps`

- `maxPlayers` : 8 slots face à 100 joueurs, c'est court — 12 à 16 sur une
  grande carte. **Change la shape de `players` (K, 16), donc un modèle
  entraîné avec K=8 ne se recharge pas.** Idem pour `grid`.
- `maxSteps` : 400 pas sur une carte 50× plus grande tronque l'épisode avant
  que quoi que ce soit d'intéressant arrive. 600–800 minimum.

### 3. Le mur d'exploration

C'est la vraie limite. Baseline aléatoire masquée sur `world`, 6 à 8 épisodes :

| adversaires | difficulté | retour | survie | durée | tuiles finales | rejets/ép. |
|---|---|---|---|---|---|---|
| 102 | Medium | −2.21 | **0/6** | 102 pas | 0 | 22 |
| 40 | Medium | −3.28 | **0/6** | 272 pas | 0 | 73 |
| 40 | Easy | −3.36 | 1/6 | 378 pas | 74 | 90 |
| 16 | Easy | −1.61 | 4/6 | 500 pas | 6 114 | 120 |

Au-delà de ~40 adversaires, **toutes les trajectoires se ressemblent** : l'agent
meurt à chaque fois, tous les retours valent −2.3, et le gradient de politique
est nul. Un entraînement de 27 000 pas sur 102 adversaires reste collé à la
valeur de la politique aléatoire (−2.17 contre −2.21) : ce n'est pas un
problème d'hyperparamètres, il n'y a rien à apprendre.

La colonne « rejets » signale un second problème de calibration : sur une carte
avec de l'eau, la politique aléatoire accumule ~90 actions rejetées par épisode
(`BOAT/unreachable` pour l'essentiel — le masque `boat_target` sait qu'une case
est une côte ennemie, pas qu'une route maritime y mène). À `w_invalid = −0.02`
cela fait **−1.8 par épisode, soit plus que tout le terme territoire
atteignable**. Baisser `w_invalid` à ~−0.005 sur ces cartes, ou resserrer le
masque, avant de conclure quoi que ce soit d'un entraînement.

**La voie praticable est un curriculum** : commencer à 16–40 adversaires en
`Easy`, puis augmenter le nombre et la difficulté en rechargeant le modèle
(les shapes ne changent pas tant que `grid` et `maxPlayers` sont fixes).

```bash
./.venv/bin/python train.py --timesteps 120000 --envs 8 --vec pipelined \
  --map world --map-source resources --bots 15 --nations 25 --difficulty Easy \
  --grid 16 --max-players 12 --max-steps 600 --out runs/ppo_world
```

Résultat réel de cette commande (≈15 min sur 10 cœurs), évalué par `eval.py`
sur 10 épisodes et **les mêmes graines** pour les deux politiques :

| politique | retour | survie | tuiles finales | durée |
|---|---|---|---|---|
| aléatoire (masqué) | −2.70 ± 0.22 | 0 % | 0 | 168 pas |
| `ppo_world` | **+34.55** ± 79.76 | **40 %** | **27 983** | 432 pas |

La courbe d'entraînement reste plate jusqu'à ~45 000 pas puis décolle
(−3.1 → +17.7) : c'est le temps qu'il faut à la politique pour apprendre à
survivre assez longtemps pour que le terme territoire devienne non nul. Un run
arrêté à 30 000 pas aurait conclu, à tort, que ça n'apprend pas.

L'écart-type énorme (±80) est attendu : selon la graine, l'agent se fait
éliminer tôt ou conquiert l'Eurasie.

## Visualiseur web

```bash
./.venv/bin/python viewer/server.py --model runs/ppo_demo.zip
```

Ouvre `http://127.0.0.1:8420`. Bibliothèque standard uniquement — aucun
framework web à installer. Un thread fait tourner une partie et pousse une
frame par décision en Server-Sent Events (~10 Ko/frame, la carte est un PNG
indexé de 1–2 Ko).

La page montre :

- **la carte complète**, avec le **cadre jaune = la fenêtre observée par le bot**
  et la grille `G × G` de la tête d'action positionnelle ; le cadre **se
  dézoome en direct** quand l'empire grandit, et la page indique le niveau
  (`zoom 3/6 · 256 tuiles`) ;
- **la case ciblée** en blanc quand l'action est un `BUILD` ou un `BOAT_ATTACK` ;
- **« ce que voit le bot »** : une vue composite reconstruite depuis les canaux
  d'observation eux-mêmes — c'est littéralement l'entrée du réseau, pas un
  rendu de la carte ;
- **les 15 canaux** un par un, en niveaux de gris ;
- l'action courante, la récompense, les compteurs, le nombre d'actions légales
  par tête, et le classement des joueurs avec leur slot `target_slot`.

Contrôles : pause / reprise, nouvelle partie, curseur de vitesse.

**La config vient du modèle.** `train.py` écrit un `<modèle>.config.json` à côté
du checkpoint, et le visualiseur comme `eval.py` le relisent : carte,
adversaires, difficulté, `grid`, `maxPlayers` — tout est restauré. Les shapes
d'observation sont en plus relues dans le checkpoint lui-même, donc un modèle
sans sidecar reste utilisable, et un drapeau contradictoire est refusé avec un
message qui nomme le drapeau à retirer plutôt qu'une erreur de shape SB3.

| Option | Défaut | Note |
|---|---|---|
| `--model` | — | un `.zip` MaskablePPO ; sans lui, politique aléatoire masquée |
| `--window-tiles` | `adaptive` | le cadre se dézoome en direct et la page affiche le niveau de zoom |
| `--grid`, `--max-players`, `--map`, `--bots`, `--nations`, `--difficulty` | lus dans le modèle | ne les passez que pour écraser |
| `--render-max-dim` | 384 | côté max de l'image envoyée au navigateur |
| `--port` | 8420 | |

Côté simulation, ce rendu n'est produit que si l'env est créé avec
`emitRender: True` — l'entraînement ne le paie jamais.

## Entraînement

```bash
./.venv/bin/python train.py --timesteps 2000000 --envs 8 --grid 16
```

Chaque env parallèle est son propre process Node, et `--vec pipelined` (le
défaut) les fait tourner simultanément. Évaluer ensuite contre la baseline,
sur les mêmes graines :

```bash
./.venv/bin/python eval.py --model runs/ppo_openfront.zip --episodes 10
```

Inutile de retaper les drapeaux de la carte : ils sont lus dans le
`<modèle>.config.json` écrit à côté du checkpoint.

## Entraîner une nouvelle version

### Dans le notebook, ou en ligne de commande ?

Le notebook convient pour **explorer** : changez `CONFIG`, relancez les cellules
à partir de la création de l'env, regardez ce que ça donne. C'est fait pour ça.

Il ne convient pas pour **produire un modèle** : il bloque le kernel pendant
tout le run, il est câblé sur 4 envs et 20 000 pas, et il n'écrit pas le
sidecar de config que le visualiseur et `eval.py` relisent. Pour un vrai
entraînement, `train.py` :

```bash
./.venv/bin/python train.py --timesteps 300000 --envs 8 \
  --map world --map-source resources --bots 15 --nations 25 --difficulty Easy \
  --grid 16 --max-players 12 --max-steps 600 \
  --out runs/ma_version
```

Puis, sans retaper aucun drapeau :

```bash
./.venv/bin/python eval.py    --model runs/ma_version.zip --episodes 10
./.venv/bin/python viewer/server.py --model runs/ma_version.zip
```

### Quels paramètres imposent de repartir de zéro

| Paramètre | Effet |
|---|---|
| `grid`, `maxPlayers` | **changent la forme de l'observation et des têtes d'action** — le modèle précédent est inutilisable, entraînement neuf obligatoire |
| tout ce qui touche `bridge/src/spec.ts` (canaux, types d'action, unités, ratios, scalaires) | idem |
| `map`, `bots`, `nations`, `difficulty` | rien à changer côté réseau |
| `maxSteps`, `ticksPerStep` | rien à changer côté réseau |
| `rewardWeights`, `tilesRewardMode` | rien à changer côté réseau |
| `windowTiles`, `windowMinTiles`, `windowZoom` | rien à changer côté réseau |
| `--lr`, `--ent-coef`, `--n-steps`, `--batch-size` | rien à changer côté réseau |

### Repartir d'un modèle existant

Tant que `grid` et `maxPlayers` ne bougent pas, `--init-from` reprend les poids
et applique la nouvelle config. C'est comme ça qu'on fait un curriculum :

```bash
# 1. apprendre les bases contre peu d'adversaires
./.venv/bin/python train.py --timesteps 150000 --envs 8 \
  --map world --map-source resources --bots 6 --nations 10 --difficulty Easy \
  --grid 16 --max-players 12 --max-steps 600 --out runs/etape1

# 2. durcir, en repartant de l'étape 1
./.venv/bin/python train.py --timesteps 150000 --envs 8 --init-from runs/etape1.zip \
  --map world --map-source resources --bots 15 --nations 25 --difficulty Easy \
  --grid 16 --max-players 12 --max-steps 600 --out runs/etape2

# 3. difficulté réelle
./.venv/bin/python train.py --timesteps 300000 --envs 8 --init-from runs/etape2.zip \
  --map world --map-source resources --bots 30 --nations default --difficulty Medium \
  --grid 16 --max-players 12 --max-steps 800 --out runs/etape3
```

### Avant de conclure qu'un run a échoué

- **Mesurez d'abord la baseline** : `eval.py` sans `--model` donne le niveau de
  la politique aléatoire masquée sur la même config. Un retour négatif ne veut
  rien dire tant qu'on ne sait pas ce que vaut le hasard.
- **Ne jugez pas avant ~50 000 pas.** Le run sur `world` est resté plat
  jusqu'à 45 000 pas puis est monté de −3.1 à +17.7. Arrêté à 30 000, il aurait
  paru mort.
- **10 épisodes minimum à l'évaluation.** L'écart-type des retours est de
  l'ordre de ±80 : sur 4 épisodes le même modèle affiche 100 % de survie, sur
  10 il en affiche 40 %.

## Performance mesurée

| Carte | Taille | Adversaires | 1 env | 8 envs (`pipelined`) |
|---|---|---|---|---|
| `big_plains` | 200×200 | 6 bots | ~1600 pas/s | — (jamais le goulot) |
| `world` | 2000×1000 | 30 bots + 72 nations | 83 pas/s | 300 pas/s |

Sur une petite carte le goulot est le réseau de neurones ; sur une grande, la
simulation. Un process Node occupe ~56 Mo, même sur 2000×1000 — la mémoire
n'est jamais la contrainte.

## Reproductibilité

`env.reset(seed=n)` désigne **la même partie dans tous les processus** : même
carte, même spawn, mêmes adversaires. C'est ce qui rend `eval.py` honnête —
les deux politiques jouent les mêmes parties, pas deux tirages différents.

Sans graine explicite, chaque env reçoit un préfixe unique (pid + compteur) de
façon que deux workers ne rejouent pas la même partie, ce qui gâcherait tout
le parallélisme.

## Étendre l'espace d'action

Trois endroits, dans cet ordre :

1. `bridge/src/spec.ts` — ajouter le nom dans `ACTION_TYPES` (et la constante
   d'index).
2. `bridge/src/actions.ts` — `ActionCodec.decode` construit l'intent,
   `ActionCodec.masks` calcule sa disponibilité.
3. Rien côté Python : les shapes et les libellés voyagent dans la frame `spec`,
   donc l'espace Gym se met à jour tout seul.

Les 24 intents du jeu sont déclarés dans
`OpenFrontIO/src/core/Schemas.ts:612-790` — alliances, donations, embargos,
nucléaire, déplacement de navires de guerre, retraite.

## Protocole du pont

Commandes montantes : JSON par ligne sur stdin
(`init`, `spec`, `reset`, `step`, `close`).
Réponses descendantes : trames binaires sur stdout,

```
[uint32 jsonLen LE][uint32 binLen LE][json utf8][payload]
```

Le header JSON décrit dtype/shape/offset de chaque tableau, reconstruits côté
Python par `np.frombuffer`. Les logs du jeu sont redirigés vers stderr pour ne
jamais polluer le flux binaire.
