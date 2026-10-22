# game_bot — faire jouer le modèle dans le vrai client OpenFront

**Statut : étape 1 — fondation.** Ce dossier contient le squelette pour piloter
un bot via le vrai protocole réseau d'OpenFront, pour que la partie soit
visible dans le vrai client (Pixi/WebGL + UI Lit). C'est destiné à filmer une
démo, pas à faire tourner un entraînement — pour entraîner, voir `rl/README.md`.

## L'idée

Le vrai jeu, c'est un serveur Node qui **relaye des intents** et une flopée de
clients qui font chacun **tourner leur propre simulation** (`src/core`), tous
pilotés par la même séquence d'intents (déterminisme). Pour qu'un modèle
piloté par Python apparaisse dans un navigateur qui regarde la partie :

1. **Serveur local** : `npm run dev` dans `OpenFrontIO/`. Il expose un endpoint
   admin HTTP (`/api/adminbot/*`) protégé par une clé — la variable
   `ADMIN_BOT_API_KEY` est déjà fixée en mode dev.
2. **Création de partie** : on `POST /api/adminbot/create_game` pour obtenir
   un `gameID` privé configuré comme on veut (carte, bots, nations…).
3. **Bot joueur** : un client WebSocket Node se connecte au worker,
   envoie un message `join` avec un token UUID anonyme, reçoit les
   `Turn`s, maintient sa simulation locale avec le vrai `GameRunner`
   d'OpenFront (import direct), et renvoie ses `Intent`s à chaque pas.
4. **Modèle Python** : branché au bot par stdin/stdout (même pattern que
   `rl/bridge`), le modèle entraîné décide des actions, le bot les convertit
   en intents et les pousse sur le WebSocket.
5. **Spectateur navigateur** : on ouvre `http://localhost:3000` dans Chrome,
   on rejoint la même partie en spectateur. On voit le bot jouer dans le vrai
   client, avec le vrai rendu — c'est l'image qu'on filme.

## Setup

Avant la toute première exécution, lier les deps d'OpenFrontIO :

```bash
cd rl/game_bot
ln -sfn ../../OpenFrontIO/node_modules node_modules
```

C'est pour que `tsx` trouve `ws`, `zod`, etc. On ne duplique rien.

## État des fichiers

| Fichier | Rôle | Statut |
|---|---|---|
| `src/create_game.ts` | crée une partie privée via l'API admin HTTP | **✓ marche** |
| `src/bot_client.ts` | pilote complet : WebSocket + sim locale + Python + intents | **✓ écrit (étape 2b)** |
| `src/local_sim.ts` | sim locale synchro, construite via `createGameRunner` | **✓ écrit** |
| `src/python_bridge.ts` | stdio ↔ modèle Python | **✓ écrit** |
| `drive.py` | charge le modèle, prédit depuis stdin, imprime action sur stdout | **✓ marche** |
| `src/test_bridge.ts` | smoke test du pont Python↔Node | **✓ marche** |
| `../viewer_demo.sh` | lance serveur + bot + ouvre navigateur spectateur | TODO étape 3 |

## Étape 1 — vérifier le pipeline WebSocket

But : prouver que l'on sait créer une partie et que l'on sait s'y connecter
avec le bon protocole. À ce stade le bot n'envoie que des `NOOP` ou pas
d'intent du tout, et la partie ne se passera pas vraiment, mais on verra
dans le navigateur spectateur que le bot est bien listé dans la lobby.

```bash
# Terminal 1 — serveur dev (3000 = vite+master, 3001 = worker 0, 3002 = worker 1)
cd OpenFrontIO && npm run dev

# Terminal 2 — notre bot
cd rl/game_bot
GAME_ID=$(../../OpenFrontIO/node_modules/.bin/tsx src/create_game.ts)
echo "gameID = $GAME_ID"
../../OpenFrontIO/node_modules/.bin/tsx src/bot_client.ts $GAME_ID

# Navigateur
# ouvrir http://localhost:9000/game/$GAME_ID
```

**Ce qu'on devrait voir à l'étape 1 :**
- `create_game.ts` imprime un `gameID` (ex: `abc12345`) et un `worker=0`.
- `bot_client.ts` imprime `socket open — sending join`, puis l'enchaînement
  des frames serveur : `lobby_info`, `prestart`, `start`, puis des `TURN N`.
- Dans le navigateur, l'URL `#/join/<gameID>` ouvre la lobby et le bot y
  figure comme joueur.

## Étape 2a — vérifier le pont Python↔Node

```bash
cd rl/game_bot
../../OpenFrontIO/node_modules/.bin/tsx src/test_bridge.ts \
    ../runs/ppo_demo.zip --grid 16 --max-players 8
```

Attendu :
```
[py] [drive] loaded ../runs/ppo_demo.zip
[test] sending zero observation...
[test] action: {"action_type":...,"target_slot":...,"ratio":...,"unit_type":...,"cell":...}
```

Si ça marche, le modèle entraîné s'exécute depuis Node et renvoie des actions.
On passera ensuite à 2b (sim locale) puis 2c (encodeurs).

## Étape 2b — bot pilote le vrai jeu

Workflow complet avec un modèle entraîné :

```bash
# Terminal 1 — serveur dev
cd OpenFrontIO && npm run dev

# Terminal 2 — crée la partie, lance le bot
cd rl/game_bot
GAME_ID=$(../../OpenFrontIO/node_modules/.bin/tsx src/create_game.ts \
    --map "Big Plains" --bots 6)
echo "ouvrir http://localhost:9000/game/$GAME_ID"

../../OpenFrontIO/node_modules/.bin/tsx src/bot_client.ts $GAME_ID \
    --model ../runs/v2_big_plains.zip \
    --grid 64 --max-players 8 --ticks-per-step 10
```

Dans Chrome (nouveau onglet) :
1. Ouvre l'URL affichée (`http://localhost:9000/game/<GAME_ID>`).
2. **Rejoins la partie comme spectateur** (sinon tu dois spawner et tu bloques la partie si tu ne cliques pas).
3. **Clique "Démarrer la partie"** si tu n'es pas spectateur, ou attends qu'un joueur humain démarre (en spectateur tu attends).

Si la lobby n'a que le bot comme joueur, la partie ne démarre pas toute seule. Deux options :
- Rejoindre le browser comme joueur, cliquer démarrer, puis spawner côté humain (le bot se place via `randomSpawn`).
- Passer `--bots` plus élevé à `create_game.ts` pour que la partie commence même sans deuxième humain.

**Les grid/max-players doivent correspondre au modèle.** `v2_big_plains.zip` → `grid=64 maxPlayers=8`.

Le bot maintient une copie locale de la simulation (via `GameRunner`,
exactement comme chaque client du jeu le fait). À chaque tick, il
sérialise l'état en observation (même encodeur que `rl/bridge`),
l'envoie au modèle Python par stdin, lit l'action, la décode en
`Intent` et la pousse sur le WebSocket.

## Étape 3 — script de démo

Un `./viewer_demo.sh --model runs/ppo_demo.zip` qui lance tout, prêt à
filmer.

## Pièges connus

- **`gitCommit`** : le serveur refuse les clients dont le `gitCommit` du
  message `join` ne correspond pas au sien. En dev c'est `"DEV"` (voir le
  script `start:server-dev` dans `OpenFrontIO/package.json`), à passer tel
  quel.
- **Déterminisme** : les deux sims (celle du bot et celle du navigateur
  spectateur) doivent voir la même séquence d'intents dans le même ordre
  pour rester synchro. C'est garanti par le serveur, qui bundle les intents
  en `Turn`s avant de les relayer.
- **zbin** : chaque frame WebSocket est un zbin binaire ; `encodeClientMessage`
  et `decodeServerMessage` dans `OpenFrontIO/src/core/ZbinWire.ts` font tout
  le travail. Pas de JSON.
