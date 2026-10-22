"""Multi-branch policy with a spatial pointer head and a player transformer.

Changements vs v1 (voir ``rl/game_bot/nn_schema_v2.pdf``) :

1. **Encodeur spatial ResNet** — l'ancien empilement de convs stride-2 est
   remplacé par 4 blocs résiduels (32→64→128→192 ch, downsampling à chaque
   bloc) ; même budget params, plus de profondeur sémantique.

2. **Décodeur spatial pour la tête `cell`** — l'ancien
   ``Linear(features_dim → G*G)`` prenait un vecteur global et prédisait un
   logit par cellule, à peu près l'inverse de ce qu'il faut faire pour une
   décision spatiale. On remplace par un décodeur U-Net style : on repart
   des features bottleneck ``(192, 8, 8)``, on module par FiLM(v) (où v est
   le vecteur global de la policy), on remonte ``(96, 16, 16) → (64, 32, 32)
   → (32, 64, 64)`` par up-convs, et on projette en 1 canal — chaque cellule
   de sortie voit sa propre région au lieu d'un vecteur plat.

3. **Transformer sur les joueurs** — l'ancien pooling mean/max/concat
   écrasait les relations entre adversaires (qui est allié, qui attaque qui,
   etc.). Remplacé par un transformer encoder (2 couches, d=128, 4 têtes)
   avec un token [CLS] appris + K tokens d'opposants. [CLS] va dans la
   fusion, les K embeddings contextualisés vont dans la tête
   target_slot via un **pointer head** (dot-product query/key).

La minimap garde son archi d'origine — la vision globale basse-résolution
n'est pas le goulot sur maps à faible proportion d'océan.

L'extracteur stocke les tenseurs latents dont la custom action head a
besoin (spatial bottleneck, slot embeddings) dans ``self._cache``, lu
synchroniquement par ``OpenFrontActionHead.forward`` dans la même passe
avant / arrière. SB3 appelle toujours l'extracteur juste avant l'action
head, donc le cache est frais.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule


# ---------------------------------------------------------------- building blocks


class ResBlock(nn.Module):
    """Pre-activation residual block with optional downsampling.

    Downsamples via stride-2 on the first conv when ``stride=2``, and a 1×1
    projection on the skip to match shapes. Kept simple on purpose — a full
    ResNet basic block with BatchNorm would need per-env stats sync in RL.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, 1, stride=stride)
        else:
            self.proj = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.proj(x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + skip)


class SpatialEncoder(nn.Module):
    """ResNet-ish encoder: ``(C, 64, 64) → (bottleneck_ch, 8, 8)``.

    Four blocks, each halving the spatial resolution and growing channels.
    """

    def __init__(self, in_ch: int, widths: tuple[int, int, int, int] = (32, 64, 128, 192)) -> None:
        super().__init__()
        c1, c2, c3, c4 = widths
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.b1 = ResBlock(c1, c1, stride=1)         # 64×64
        self.b2 = ResBlock(c1, c2, stride=2)         # 32×32
        self.b3 = ResBlock(c2, c3, stride=2)         # 16×16
        self.b4 = ResBlock(c3, c4, stride=2)         #  8×8
        self.out_ch = c4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        return x


class SpatialDecoder(nn.Module):
    """Decoder that lifts ``(bottleneck_ch, 8, 8)`` back to a ``(1, G, G)`` logit map.

    Only valid for G=64 (three up-convs ×2). FiLM-modulated by the global
    latent vector so the per-cell logits know what the agent "wants" to do;
    without this, the cell head would have no idea of the chosen
    ``action_type`` or any global context.
    """

    def __init__(self, in_ch: int, latent_dim: int, grid: int = 64) -> None:
        super().__init__()
        assert grid == 64, f"SpatialDecoder assumes G=64, got {grid}"
        self.film = nn.Linear(latent_dim, 2 * in_ch)
        self.up1 = nn.ConvTranspose2d(in_ch, 96, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.proj = nn.Conv2d(32, 1, 1)
        self.act = nn.ReLU(inplace=True)
        self.in_ch = in_ch

    def forward(self, feat: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        gb = self.film(latent)                        # (B, 2C)
        g = gb[:, : self.in_ch].unsqueeze(-1).unsqueeze(-1)
        b = gb[:, self.in_ch :].unsqueeze(-1).unsqueeze(-1)
        x = feat * (1.0 + g) + b
        x = self.act(self.up1(x))
        x = self.act(self.up2(x))
        x = self.act(self.up3(x))
        return self.proj(x).flatten(1)                # (B, G*G)


class PlayerTransformer(nn.Module):
    """Transformer encoder over the K opponent slots + a learned [CLS].

    Output:
        cls_emb   (B, d_model)   — pooled summary for the fusion branch
        slot_emb  (B, K, d_model) — per-slot contextualised embeddings for
                                     the target_slot pointer head

    Empty opponent slots are passed as padding (``key_padding_mask``), so
    they don't participate in attention. The validity flag is slot 0 of each
    row (see ``rl/bridge/src/spec.ts`` — ``PLAYER_FEATURES[0] == "valid"``).
    """

    def __init__(
        self,
        pf: int,
        k_players: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(pf, d_model)
        self.pos = nn.Parameter(torch.zeros(k_players, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        # enable_nested_tensor=False: the nested-tensor fast path is
        # incompatible with norm_first=True (pre-norm transformer); torch
        # warns about it otherwise. The slow path is fine here, K is tiny.
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.d_model = d_model
        self.k_players = k_players
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

    def forward(
        self, players: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # players: (B, K, PF). Slot 0 of each row is the validity flag.
        valid = players[..., 0].bool()                 # (B, K)
        x = self.embed(players) + self.pos             # (B, K, d)
        cls = self.cls.expand(x.size(0), -1, -1)       # (B, 1, d)
        tokens = torch.cat([cls, x], dim=1)            # (B, K+1, d)
        # key_padding_mask=True means "ignore this position". CLS is never
        # padded; opponent j is padded when valid[b,j] is False.
        pad = torch.cat(
            [torch.zeros_like(valid[:, :1]), ~valid], dim=1
        )                                              # (B, K+1)
        # When *every* slot is invalid (shouldn't happen mid-game but can
        # right at reset) the encoder would nan on an all-masked row. Guard
        # by never masking the CLS and by replacing fully-dead rows with the
        # untouched token (padded op will see only CLS).
        out = self.encoder(tokens, src_key_padding_mask=pad)
        cls_out = out[:, 0]                            # (B, d)
        slot_out = out[:, 1:]                          # (B, K, d)
        # Zero out embeddings of invalid slots so the downstream pointer
        # head cannot anchor to them. (Masking still happens in the action
        # distribution, but keeping their embeddings alive risks a learned
        # "point at an invalid slot" bias that fights the mask.)
        slot_out = slot_out * valid.unsqueeze(-1).float()
        return cls_out, slot_out


# ---------------------------------------------------------------- feature extractor


class OpenFrontExtractor(BaseFeaturesExtractor):
    """Returns a fused global feature vector and stashes side tensors.

    ``self._cache`` holds the spatial bottleneck ``(B, C, 8, 8)`` and the
    per-slot embeddings ``(B, K, d)``. The custom action head reads them in
    the same forward pass, after SB3 called ``forward`` on this extractor.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        scalar_dim: int = 128,
        minimap_dim: int = 128,
        spatial_pool_dim: int = 192,
        player_d_model: int = 128,
        spatial_widths: tuple[int, int, int, int] = (32, 64, 128, 192),
        n_player_layers: int = 2,
    ) -> None:
        fused_dim = scalar_dim + minimap_dim + spatial_pool_dim + player_d_model
        super().__init__(observation_space, fused_dim)

        c, h, w = observation_space["spatial"].shape
        mc, mh, mw = observation_space["minimap"].shape
        s = observation_space["scalar"].shape[0]
        k, pf = observation_space["players"].shape

        # ---- Spatial branch (ResNet encoder) ----
        self.spatial = SpatialEncoder(in_ch=c, widths=spatial_widths)
        self.spatial_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.spatial.out_ch, spatial_pool_dim),
            nn.ReLU(inplace=True),
        )

        # ---- Minimap branch (kept simple — not the current bottleneck) ----
        self.minimap_cnn = nn.Sequential(
            nn.Conv2d(mc, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            probe = torch.zeros(1, mc, mh, mw)
            mm_out = self.minimap_cnn(probe).flatten(1).shape[1]
        self.minimap_head = nn.Sequential(
            nn.Linear(mm_out, minimap_dim),
            nn.ReLU(inplace=True),
        )

        # ---- Scalar branch ----
        self.scalar_mlp = nn.Sequential(
            nn.Linear(s, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, scalar_dim),
            nn.ReLU(inplace=True),
        )

        # ---- Player branch (transformer) ----
        self.player_tr = PlayerTransformer(
            pf=pf,
            k_players=k,
            d_model=player_d_model,
            n_heads=4,
            n_layers=n_player_layers,
            ff_dim=player_d_model * 2,
        )

        self.n_players = k
        self.player_d_model = player_d_model
        self.spatial_bottleneck_ch = self.spatial.out_ch

        # Set by ``forward``, read by ``OpenFrontActionHead``.
        self._cache: dict[str, torch.Tensor] = {}

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        spatial_feat = self.spatial(obs["spatial"].float())        # (B, C, 8, 8)
        spatial_vec = self.spatial_pool(spatial_feat)              # (B, Ds)
        minimap_vec = self.minimap_head(
            self.minimap_cnn(obs["minimap"].float()).flatten(1)
        )                                                           # (B, Dm)
        scalar_vec = self.scalar_mlp(obs["scalar"].float())        # (B, Dsc)
        cls_vec, slot_emb = self.player_tr(obs["players"].float())  # (B, D), (B, K, D)

        self._cache = {
            "spatial_bottleneck": spatial_feat,
            "slot_emb": slot_emb,
        }
        return torch.cat([spatial_vec, minimap_vec, scalar_vec, cls_vec], dim=1)


# ---------------------------------------------------------------- action head


class OpenFrontActionHead(nn.Module):
    """Replaces the default ``Linear(latent, sum(nvec))`` action head.

    Produces the same flat logit vector in nvec order
    ``[action_type, target_slot, ratio, unit_type, cell]`` so SB3's masked
    MultiCategorical distribution can split it as usual, but:

    * ``action_type``, ``ratio``, ``unit_type`` — plain linear heads from
      ``latent_pi`` (nothing fancy is needed, they are small discrete sets).
    * ``target_slot`` — **pointer** over the K contextualised slot
      embeddings produced by the player transformer. Index 0 is
      TerraNullius (handled by a learned bias key).
    * ``cell`` — **spatial decoder** that upsamples the spatial bottleneck
      to a ``(1, G, G)`` logit map, FiLM-modulated by ``latent_pi``.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dims: list[int],
        grid: int,
        k_players: int,
        extractor: OpenFrontExtractor,
    ) -> None:
        super().__init__()
        n_at, n_slot, n_ratio, n_unit, n_cell = action_dims
        assert n_slot == k_players + 1, (
            f"target_slot nvec {n_slot} should be K+1={k_players + 1}"
        )
        assert n_cell == grid * grid, (
            f"cell nvec {n_cell} should be G*G={grid * grid}"
        )

        # The extractor is a *parameter* module of the parent policy; we only
        # keep a reference to grab its cache, not to re-register parameters.
        # (nn.Module.__setattr__ would otherwise double-register.)
        object.__setattr__(self, "_extractor", extractor)

        self.at_head = nn.Linear(latent_dim, n_at)
        self.ratio_head = nn.Linear(latent_dim, n_ratio)
        self.unit_head = nn.Linear(latent_dim, n_unit)

        d = extractor.player_d_model
        self.q_proj = nn.Linear(latent_dim, d)
        self.k_proj = nn.Linear(d, d)
        # TerraNullius key: learned, shared across the batch, matches slot
        # index 0 in the MultiDiscrete target_slot head.
        self.terra_key = nn.Parameter(torch.zeros(d))
        self.pointer_scale = d**-0.5

        self.cell_decoder = SpatialDecoder(
            in_ch=extractor.spatial_bottleneck_ch,
            latent_dim=latent_dim,
            grid=grid,
        )
        nn.init.normal_(self.terra_key, std=0.02)

    def forward(self, latent_pi: torch.Tensor) -> torch.Tensor:
        cache = self._extractor._cache
        spatial = cache["spatial_bottleneck"]          # (B, C, 8, 8)
        slot_emb = cache["slot_emb"]                   # (B, K, d)

        at_logits = self.at_head(latent_pi)
        ratio_logits = self.ratio_head(latent_pi)
        unit_logits = self.unit_head(latent_pi)

        q = self.q_proj(latent_pi) * self.pointer_scale   # (B, d)
        ks = self.k_proj(slot_emb)                        # (B, K, d)
        slot_logits = (q.unsqueeze(1) * ks).sum(-1)       # (B, K)
        terra_logit = (q * self.terra_key).sum(-1, keepdim=True)  # (B, 1)
        target_logits = torch.cat([terra_logit, slot_logits], dim=1)  # (B, K+1)

        cell_logits = self.cell_decoder(spatial, latent_pi)  # (B, G*G)

        return torch.cat(
            [at_logits, target_logits, ratio_logits, unit_logits, cell_logits],
            dim=1,
        )


# ---------------------------------------------------------------- policy


class OpenFrontPolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy wired to the custom extractor and action head.

    Overrides ``_build`` to swap SB3's default
    ``Linear(latent, sum(nvec))`` action net for ``OpenFrontActionHead``.
    Everything else — forward, evaluate_actions, log_prob, etc. — is
    inherited unchanged, because ``_get_action_dist_from_latent`` only ever
    calls ``self.action_net(latent_pi)`` and treats its output as flat
    logits in nvec order, which is exactly what our head produces.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        **kwargs: Any,
    ) -> None:
        # Force shared extractor: our action head grabs cached tensors from
        # the pi extractor; a separate vf extractor would populate the wrong
        # cache. (Share saves memory too — the biggest branch is spatial.)
        kwargs.setdefault("share_features_extractor", True)
        if not kwargs["share_features_extractor"]:
            raise ValueError(
                "OpenFrontPolicy requires share_features_extractor=True — "
                "the action head reads the pi extractor's cache."
            )
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        # Decode action layout from the MultiDiscrete nvec. The env guarantees
        # [action_type, target_slot=K+1, ratio, unit_type, cell=G*G]; we
        # recover K and G from it rather than threading them from the config.
        nvec = list(map(int, self.action_space.nvec))
        n_cell = nvec[-1]
        grid = int(round(n_cell**0.5))
        if grid * grid != n_cell:
            raise ValueError(f"cell nvec {n_cell} must be a perfect square")
        k_players = nvec[1] - 1

        self.action_net = OpenFrontActionHead(
            latent_dim=self.mlp_extractor.latent_dim_pi,
            action_dims=nvec,
            grid=grid,
            k_players=k_players,
            extractor=self.features_extractor,
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: float(np.sqrt(2)),
                self.mlp_extractor: float(np.sqrt(2)),
                self.action_net: 0.01,
                self.value_net: 1.0,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )


# ---------------------------------------------------------------- defaults


DEFAULT_POLICY_KWARGS: dict = {
    "features_extractor_class": OpenFrontExtractor,
    "features_extractor_kwargs": {},
    "net_arch": {"pi": [768, 512], "vf": [512]},
    "share_features_extractor": True,
}


__all__ = [
    "OpenFrontExtractor",
    "OpenFrontActionHead",
    "OpenFrontPolicy",
    "DEFAULT_POLICY_KWARGS",
]
