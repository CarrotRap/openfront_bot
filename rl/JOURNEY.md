# Journal — entraînement d'une IA sur OpenFront

Récit chronologique des décisions, bugs, et pivots pendant le développement
du bot RL. Garder ce fichier à jour quand on fait un pivot technique
important — sert aussi de brouillon pour le script vidéo.

Chaque entrée : **date · ce qu'on a changé · pourquoi · impact observé.**

---

## Phase 1 — Mise en place (avant 2026-10-14)

- Stack construit : `rl/bridge/` (sim headless TypeScript), `rl/openfront_gym/`
  (env Gymnasium), `train.py` (MaskablePPO), `rl/game_bot/` (client WebSocket
  pour jouer dans la vraie game).
- Architecture du réseau : multi-branche, 4.5M params. CNN tower sur la vue
  égocentrée (26+K canaux × 64 × 64), mini-CNN sur la minimap (5 × 32 × 32),
  MLP scalaire (46 réels), MLP partagé sur les K slots d'adversaires. Fusion
  → 768 features → 5 têtes d'action (type, slot, ratio, unit, cell) + value.
- Reward v1 (defaults) :
  ```
  tiles=1.0, troops=0.05, gold=0.02 (sur Δlog du solde),
  invalid=-0.02, death=-1, win=+5, step=-0.0005
  ```

## Phase 2 — Premier modèle (2026-10-14)

- Base : `v2_big_plains` entraîné sur map `big_plains`, nations=disabled.
- Fine-tune World/Easy, LR aggressif → KL=0.07 à iter 4, policy dérive.
- v4_world_long : LR 1e-4, converge à **ep_rew ≈ 42-50** sur nations=16/Easy.
- **Choc visuel** : en démo le bot tapote frénétiquement ses frontières au
  lieu de conquérir. **Le reward dit "bien", l'image dit "nul".**

## Phase 3 — Diagnostic du reward hacking (2026-10-14)

Analyse de la politique optimale sous reward v1 :
- Gratter 1 tile = +1, zéro risque de contre-attaque.
- Mourir à mi-partie = seulement -1, négligeable face aux +45 accumulés.
- Win = +5, mais max_steps=800 (13 min game-time) → win ne tombe JAMAIS.

→ La policy a trouvé le local optimum : **nibbling + survivre tant pis**.

## Phase 4 — Reward v2 (2026-10-14)

Fichier : `rl/bridge/src/env.ts`

```diff
- tiles=1.0            (symétrique)
+ tilesGain=1.0, tilesLoss=3.0   (loss aversion)
- gold sur Δlog(solde)
+ gold sur log(1+income)         (ne punit plus les achats)
- death=-1
+ death=-5
- win=+5
+ win=+50
+ elim=+2 par KO
+ max_steps 800 → 3000 (côté train.py)
```

Résultat sur v7_reward_v2 (nations=16/Easy, init de v5) :
- Iter 1-5 : ep_rew chute à **-3** (choc de distribution).
- Iter 10-30 : récupération à **+70**.
- Iter 50 : plateau à **ep_rew=140, ep_len=1120** (survie 2× plus longue).
- **Visuellement** : bot qui consolide avant d'attaquer, ne grignote plus.

## Phase 5 — Spawn picker (2026-10-15)

Fichier : `rl/game_bot/src/spawn_picker.ts`

Heuristique codée à la main (pas du RL) : au début de la partie, échantillonne
500 tiles au hasard, filtre (land, non-owned, non-border), choisit celle qui
maximise la distance Manhattan min aux autres joueurs déjà spawnés. Re-pick
toutes les 5 ticks (~500ms) pendant les 20s de spawn phase pour réagir si
un humain bouge son spawn.

Côté serveur, c'était possible gratuitement : `SpawnExecution` relinquit les
anciennes tiles et reconquiert autour du nouveau centre, le tout
déterministiquement sur tous les clients.

→ Angle vidéo : **"L'IA fait le gros du boulot, mais pour une règle qu'un
joueur humain connaît par cœur ("spawn loin"), autant la coder à la main."**

## Phase 6 — Curriculum nations=25/Medium (2026-10-15)

- v6 (init depuis v5-reward-v1) : chute à **ep_rew=-3**, bot meurt en 3min.
  → Il n'avait pas appris à survivre, loss aversion v2 pas transférée.
- v9 (init depuis v8-reward-v2) : converge à **ep_rew=66, ep_len=333**.
  → Delta +70 de reward à iter équivalente juste parce que v2 a appris
     la préservation. **Démonstration directe de la valeur du curriculum.**

## Phase 7 — Scaling à nations=50/Easy (2026-10-15)

v10 (init depuis v9) :
- Iter 1 : ep_rew=11
- Iter 36 : **ep_rew=269, ep_len=1540 (~26 min de game time)**
- Policy et critic sains. value_loss volatile (épisodes longs → variance
  intrinsèque élevée) mais explained_variance moyen à 0.8.

## Phase 8 — Les bugs de câblage (2026-10-16)

Le bot v10 **avait l'air nul en démo**. On a cru à un plafond du modèle.
En fait trois bugs de plomberie empilés :

### 8.1 — `maxSteps` hardcodé à 1200 dans bot_client.ts

Fichier : `rl/game_bot/src/bot_client.ts`

Les scalars `ticks / (maxSteps × ticksPerStep)` et `step / maxSteps` sont dans
l'observation. Avec `maxSteps=1200` en live vs `maxSteps=3000` en training,
**au step 300 le bot croyait être à 25% du jeu au lieu de 10%**. Il passait
en mode "end game : envoyer tout partout" dès les 3 premières minutes.

Fix : charger le sidecar `.config.json` à côté du `.zip` et appliquer ses
valeurs à la place des defaults hardcodés. Shape-defining fields (grid,
maxPlayers) → crash fort si mismatch.

### 8.2 — Champ `nations` ignoré par le serveur

Fichier : `rl/game_bot/src/create_game.ts`

Historiquement on envoyait `disableNPCs: args.nations === "disabled"` au
serveur. **Mais `disableNPCs` n'existe pas dans `GameConfigSchema`**. Zod
le strip silencieusement, et `GameManager.createGame` remplit le champ
manquant par `"default"` → **toutes les 72 nations du manifest World
spawnent, quel que soit le --nations CLI**.

Fix : envoyer le vrai champ `nations` (uint 1-400 | "default" | "disabled").

Impact : avant → démo contre 79 joueurs. Après → démo contre 57 joueurs
(matche l'entraînement v10).

### 8.3 — `deterministic=True` collapse sur NOOP

Fichier : `rl/game_bot/drive.py`

En argmax, l'action la plus probable sur chaque tête est choisie. Sur un
espace d'actions large (4 types × 9 slots × 6 ratios × 7 units × 4096 cells),
NOOP (seul choix passif) absorbe ~20-25% de la masse, tandis que chaque
action concrète est à ~5%. Argmax → NOOP. Toujours.

Fixes implémentés :
- `--mask-noop` : masque le NOOP dans l'action_type (sauf si c'est le seul
  valide), force une action concrète.
- `--deterministic` : argmax vs sampling, configurable.
- Protection : si TOUTES les non-NOOP sont masquées, on remet NOOP (évite
  distribution vide).

Dans demo.sh : `DETERMINISTIC=1` par défaut, c'est tout.

## Phase 9 — Reward v3 — hold reward (2026-10-16)

Fichier : `rl/bridge/src/env.ts`

Nouvelle hypothèse après observation : avec reward v2, un bot qui ne fait
rien dans un coin tranquille **touche un reward positif ou nul**. Loss
aversion × 3 rend attaquer risqué (contre-attaque possible). Troops/gold
accumulent passivement. **La passivité est devenue un local optimum.**

Ajout du terme `hold` :

```
+ w.hold × (tiles / tilesReference)   // par step
```

Avec `w.hold = 0.001` et tilesRef ≈ 158 (World 57 joueurs) :
- 100 tiles sur 1500 steps = +0.95 (passivité : négligeable)
- 1000 tiles sur 1500 steps = +9.5
- 2000 tiles sur 1500 steps = +19
- Full map sur 2500 steps = +1700 (incentive fort à conquérir)

Prochain run : v11_reward_v3 init depuis v10, même config, pour isoler
l'effet du reward v3 vs v2.

## Phase 9b — Reward v3.1 (2026-10-16, après observation v11)

Après ~27k steps de v11, le bot avait appris à **prendre tout le terrain
neutre** (zéro risque, grab trivial) puis **NOOP en boucle** sans jamais
attaquer un vrai joueur. Comportement rationnel face au reward :

| Scenario | Reward avec v3 (hold=0.001) |
|---|---|
| Gain 100 tiles sur joueur (one-shot) | +0.63 |
| Hold cumulé si on garde | +0.63 |
| Loss 50 tiles sur contre (×3) | -0.95 |
| Hold perdu | -0.32 |
| **EV attaque** | **≈ 0** |
| **EV NOOP** | **0 avec variance 0** |

→ NOOP strictement préférable (même EV, variance plus basse).

Fix v3.1 (deux leviers simultanés) :
- `w.hold: 0.001 → 0.005` (5× stronger pull vers "être grand")
- `w.tilesLoss: 3.0 → 2.0` (moins de pénalité pour contre-attaque ; le hold
  punit déjà les pertes implicitement)

Avec hold=0.005 le même scénario donne EV ≈ +0.63 pour l'attaque — clairement
positive.

## Phase 10 — Reward v4-neutre (2026-10-17)

Trois symptômes observés sur v11, v12, v13 (tous reward v3.x avec hold linéaire) :
1. **Spike + regression** sur ep_rew (pic transient à 1150-963, retour à 300-500)
2. **ep_len figé à ~640** (le bot ne gagne pas en survie)
3. **Variance explosive** du reward entre épisodes de domination et épisodes
   moyens (facteur 30× → gradient instable)

Cause structurelle : **le hold linéaire** récompensait de façon illimitée les
grands empires, créant des outliers que la policy pousse à reproduire mais
qui n'étaient que de la chance. Les updates divergent.

**Discussion stratégique** : l'utilisateur a décrit sa stratégie humaine en
5 phases (grab neutres → finir le faible → économie → prédation opportuniste
→ alliances). Deux options d'encodage :
- **v4-biased** : encoder chaque phase comme un sous-reward (neutralGain,
  structBuilt, etc.) → converge vite mais plafonne au skill humain
- **v4-neutre** : donner juste la condition de victoire et laisser la policy
  découvrir ses propres phases → plus lent mais potentiel super-humain

Choix pour la vidéo : **v4-neutre**. Angle narratif cohérent avec AlphaGo
Zero ("l'IA réinvente le jeu toute seule"), honnête sur le trade-off, et
évite le piège de l'anchoring aux biais humains.

Reward v4-neutre :

```typescript
tilesGain: 0.5    // dense signal, pas une stratégie
tilesLoss: 0.5    // symétrique maintenant, pas de loss aversion forte
troops: 0.05      // inchangé
gold: 0.02        // income (inchangé)
invalid: -0.01    // léger
elim: 15.0        // ×7.5 vs v3.1 — tuer un adversaire est une accélération
                  //   objective vers la win condition (80% des tiles)
death: -15.0      // ×3 vs v3.1 — survivre est critique
win: 100.0        // ×2 vs v3.1 — l'objectif réel
step: 0           // supprimé : progress encode déjà "temps qui passe = raté"
progress: 2.0     // NOUVEAU — +2 × min(1, tiles / tilesToWin) par step
                  //   tilesToWin = 0.80 × land (PERCENT_TILES_OWNED_TO_WIN
                  //   dans Config.ts). Capped à 1 → pas de runaway.
```

**Pas de `hold`** — remplacé par `progress` plafonné à la win condition.
**Pas de `neutralGain`, `structBuilt`** — on reste neutre sur la *méthode*.

Attendu : la variance de reward entre épisodes converge (plus de spike),
ep_len devrait remonter (survie récompensée directement par survival ×
progress), et la policy explore sa propre stratégie sans copier la mienne.

Prochain run : v14 init depuis v10, nations=50/Easy.

## Phase 11 — Deux bugs de câblage critiques trouvés APRÈS v14 (2026-10-17)

v14 a convergé proprement (ep_rew=1800, ep_len=1500) mais **joue horriblement
mal en démo**. L'utilisateur a demandé "y a pas un bug?". Oui. Deux :

### Bug A — Scalar goldIncome passait le CUMULATIF au lieu du DELTA

Fichier : `rl/game_bot/src/bot_client.ts`

Training (env.ts:388, 460) :
```typescript
const income = Math.max(0, goldEarned - this.prevGoldEarned);  // Δ par step
this.encoder.encode(this.me, this.step_, income);
```

Live (avant fix) :
```typescript
const goldIncome = Number(me.goldEarned()); // CUMULATIF
encoder.encode(me, assets.step, goldIncome);
```

Impact : le scalar #7 `clamp01(Math.log1p(goldIncome) / 20)` :
- Training : income par step ≈ 0-200 → scalar ≈ 0.1-0.27
- Live : cumulatif 10000+ → scalar saturé à 1.0 au bout de quelques minutes

Fix : tracker `prevGoldEarned` et calculer le delta comme en training.

### Bug B — Les 7 flags `can_build_*` dans l'obs étaient tous à ZÉRO

Fichier : `rl/game_bot/src/bot_client.ts`

Training (env.ts:461-464) :
```typescript
const masks = this.codec.masks(this.me, obs, obs.scalar.subarray(CAN_BUILD_OFFSET));
// codec écrit DANS obs.scalar — les 7 derniers slots remplis in-place
```

Live (avant fix) :
```typescript
const dummyCanBuild = new Float32Array(7);  // buffer séparé jeté
const masks = codec.masks(me, enc, dummyCanBuild);
```

Impact : la policy voyait **can_build_city=0, can_build_port=0, can_build_defense_post=0, ...**
pour les 7 unit types, à chaque step. Elle "croyait" ne jamais pouvoir
construire au niveau scalar. Même avec de l'or infini en poche, le scalar
disait "non". Comportement complètement cassé en déploiement.

Fix : passer `enc.scalar.subarray(CAN_BUILD_OFFSET)` au lieu d'un dummy,
pour que codec.masks écrive dans la vraie obs.

### Pour la vidéo

**C'est un chapitre majeur à part entière**. Angle narratif :

*"Après des heures d'entraînement, mes métriques étaient magnifiques. Le bot
avait appris, j'étais fier. Je lance la démo — nul. J'ai cru que c'était un
problème d'entraînement. Pendant deux jours j'ai tweaké le reward. Rien ne
marchait. Finalement j'ai découvert que mon script de déploiement envoyait
au modèle des observations SUBTILEMENT différentes de celles qu'il avait vues
pendant l'entraînement. Un champ de l'observation était calculé différemment,
un autre était à zéro au lieu d'être rempli. Mon IA n'était pas nulle — elle
voyait un monde qui n'existait pas."*

C'est **LE problème classique du MLOps** : mismatch training/serving. Google
a des équipes entières là-dessus (TFX, Vertex AI). Pour un side project ça
m'a pris 3 jours.

Prochain test : relancer `demo.sh` avec v14 et les deux bugs fixés. **Pas
besoin de retrainer** — le modèle est bon, c'était l'input qui mentait.

---

## Angles narratifs pour la vidéo

- **Chapitre pivot** : *"Métrique qui monte ≠ bot qui joue bien."* Le vrai
  boulot ML c'est pas de construire le réseau, c'est de **lui expliquer
  ce que je veux**.
- **La passivité** : *"J'ai découvert que mon IA préférait ne rien faire
  plutôt que risquer un mauvais coup. Comme un stagiaire qui ne propose
  jamais rien plutôt que proposer une bêtise."*
- **Les bugs silencieux** : *"Mon script envoyait un champ qui n'existait
  pas. Le serveur l'ignorait silencieusement, et jouait avec les valeurs
  par défaut. Mon bot entraîné à 16 nations se retrouvait face à 72."*
- **Mismatch training/déploiement** : *"Une seule constante hardcodée
  faisait croire au modèle qu'on était en fin de partie dès les 3 premières
  minutes. Il se comportait comme un joueur qui panique à 10 secondes du
  compte à rebours — alors qu'il restait 25 minutes."*

---

## Phase 12 — v14 démo post-fix : « bon mais pas excellent » (2026-10-18)

Démo relancée avec v14 et les deux bugs câblés corrigés. Verdict de l'utilisateur :
**« ok il est bon mais pas excellent du tout »**. Pas de retrain nécessaire — le
modèle est propre, c'est le comportement final qui plafonne.

Observations :
- Le bot s'étend correctement sur les neutres (progress reward fait son taf)
- Il attaque les voisins mais sans toujours finir
- Pas de spam BOAT absurde (reward v4-neutre a bien tué le travers de v2)
- Mais il ne « domine » pas visuellement comme un bon humain le ferait

Deux hypothèses pour la vidéo :
- **Compétence réelle** : la policy a plafonné sur Easy. Trop simple = pas assez
  de pression pour pousser plus loin.
- **Limitation de masking** : les 5 têtes sont indépendantes (joint-mask non
  implémenté). Argmax/sampling peut produire des combos illégaux (ex. ATTACK
  + cell sans voisinage), rejetés par l'engine → la policy perd des tours.

Décision : **monter la difficulté** plutôt que patcher le masking. Le vrai test
d'une IA RL c'est l'adversité. Si elle a appris sur Easy, elle doit apprendre
plus sur Medium.

Flag chaud : le joint-mask (~40 lignes dans `drive.py` pour filtrer les cells
compatibles avec le `action_type` avant argmax/sampling) reste disponible si
après Medium le comportement est encore limite.

## Phase 13 — v15 Medium : l'adversité tape fort (2026-10-18)

Curriculum : init de `v14_reward_v4_neutral.zip`, **nations=50, difficulty=Medium**,
200k steps. L'intuition : Medium = adversaires qui contre-attaquent vraiment →
la policy doit soit tuer vite soit mourir.

**Résultat partiel (iter 40/65, 61% du run) :**

| | v14 (Easy) final | v15 (Medium) iter 40 |
|---|---|---|
| `ep_rew_mean` | 1700 | 468 |
| `ep_len_mean` | 1200 | 310 |
| `rew / step` | 1.4 | 1.5 |

Lecture : **même efficacité par step, 4× moins de temps de survie**. Les bots
Medium la tuent avant qu'elle ait fini de dominer. Métriques PPO saines (KL
0.006-0.010, clip ~0.05-0.09, EV 0.85-0.93) → fine-tune propre, pas de
divergence, juste un nouveau plafond.

C'est **le même modèle, dans un monde différent**. La policy n'est pas
« moins bonne » — elle est capped par l'environnement.

### Pour la vidéo

Angle narratif qu'on tient là, assez rare dans les vidéos ML grand public :

*"Je monte la difficulté. Mes métriques s'effondrent — reward divisé par 4,
survie divisée par 4. Mais regardez bien : par step, la policy est aussi
efficace. Elle n'est pas devenue nulle. Elle vit dans un monde plus dur. C'est
la même IA, mais face à des adversaires qui ne la laissent plus respirer. En
machine learning, les chiffres ne veulent rien dire sans le contexte."*

C'est le pendant parfait de la Phase 10 (bugs câblage) : d'un côté « métriques
belles mais bot cassé » (training/serving mismatch), de l'autre « métriques
en chute mais bot sain » (difficulté qui plafonne la perf absolue).

## Phase 14 — Refonte architecturale (2026-10-19)

Fichier : `rl/openfront_gym/policy.py`, schéma : `rl/game_bot/nn_schema_v2.pdf`.

Après Phase 13 le constat devient : le bot plafonne sur Easy ET Medium, et
l'archi v1 (4.5M params, 3 convs stride-2 + mean/max/concat sur joueurs +
`Linear(768→4096)` pour la tête `cell`) a des faiblesses structurelles qui
ne se résolvent pas en ajoutant du reward ou du temps.

Trois changements :

### 14.1 — Décodeur spatial pour la tête `cell`

L'ancien head était `Linear(latent → G²)`. **Pour une décision spatiale (quelle
tile ?), c'est l'inverse de ce qu'il faut** : le réseau extrait des features
spatiales via CNN, puis les flatten → MLP → logits plats, perdant toute
information de position. Chaque cellule de sortie ne voit qu'un vecteur global.

Nouveau : décodeur U-Net style qui remonte du bottleneck `(192, 8, 8)` vers
`(1, 64, 64)` via 3 up-convs, FiLM-modulé par `latent_pi` (le vecteur policy
global injecte le contexte "quelle action, quel adversaire"). Chaque pixel
de sortie voit sa propre région.

### 14.2 — Encodeur spatial ResNet

Remplacement des 3 convs stride-2 par 4 blocs résiduels pre-activation
(32→64→128→192 ch, downsampling à chaque bloc). Même ordre de grandeur en
params, mais plus de profondeur et un gradient qui circule mieux.

### 14.3 — Transformer sur les joueurs + pointer head `target_slot`

Ancien pooling mean/max/concat écrasait toute relation (qui est allié, qui
attaque qui, qui est traître). Remplacé par un transformer encoder (2 couches,
d=128, 4 têtes) sur K+1 tokens : un [CLS] appris + K adversaires. Slots vides
ignorés via `key_padding_mask`.

Sortie : [CLS] contextualisé pour la fusion, K embeddings pour la **pointer
head** `target_slot` (dot-product `<query(latent_pi), key(slot_emb)>` + une
clé apprise pour TerraNullius à l'index 0).

### Bilan

Params : 4.5M → **3.4M** (plus petit que v1, mais structure massivement meilleure).

```
features_extractor  1,895,200   (ResNet spatial + minimap CNN + scalar MLP + transformer joueurs)
mlp_extractor       1,132,288   (pi=[768,512], vf=[512])
action_net            394,738   (linear heads + pointer + spatial decoder)
value_net                 513
```

Pas de reprise possible depuis v14/v15 (shapes incompatibles → entraînement
from-scratch). Smoke test PPO end-to-end sur Montreal : 128 steps, converge
proprement, round-trip load+predict OK via le chemin de `drive.py`.

Prochain run : v16 from scratch sur **Montreal** (85% land — pas de goulot
"vision globale océan" pour isoler l'effet des changements archi).

```bash
python train.py --map Montreal --map-source resources \
  --nations 25 --difficulty Easy --envs 8 \
  --timesteps 500000 --out runs/v16_montreal_newarch
```

### Pour la vidéo

Chapitre « AlphaStar en miniature » : trois idées des bots pro RTS
transplantées dans mon réseau — décision spatiale faite spatialement, joueurs
traités comme un ensemble avec relations, et une tête `cell` qui voit vraiment
la carte au lieu d'un vecteur plat.

---

## État actuel

- **Modèle de référence (démo propre)** : `runs/v14_reward_v4_neutral.zip`
  (nations=50/Easy, reward v4-neutre, ep_rew=1700, ep_len=1200). Les deux bugs
  câblage sont corrigés dans `bot_client.ts`.
- **En entraînement** : `v15_medium.zip` (init v14, nations=50/Medium, 200k
  steps, iter 40/65 au moment où j'écris).
- **Démo** : `./demo.sh --model ../runs/v14_reward_v4_neutral.zip` (ou v15
  quand il finit). Sampling par défaut (DETERMINISTIC=0) pour éviter la
  collapse argmax→NOOP sur gros joint action space.
- **Flags critiques en live** :
  - sidecar `.config.json` lu automatiquement
  - spawn picker actif (randomSpawn=false)
  - nations=50 par défaut (matche l'entraînement)
  - goldIncome passé en **delta** (fix bug A)
  - `can_build_*` scalars écrits dans la vraie obs (fix bug B)
- **TODO potentiel** : joint-mask dans `drive.py` (filter cells par action_type
  avant sampling) — ~40 lignes, seulement si après v15 le bot reste capped par
  des actions illégales.
