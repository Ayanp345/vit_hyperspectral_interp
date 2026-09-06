# Mechanistic Interpretability of Hyperspectral Vision Transformers

A Vision Transformer trained to classify hyperspectral scenes (135 bands,
Pixxel Firefly-like), with a real mechanistic-interpretability toolkit —
Logit Lens, Attention Rollout, and gradient-based spectral band
attribution — to answer: *which narrow spectral bands is the network
actually using, and at which layer does it commit to its answer?*

## Two execution tracks, same honesty policy as the earlier SAE project

| Track | Where | Why |
|---|---|---|
| `colab/hyperspectral_vit_colab.ipynb` | **Google Colab** | Real PyTorch: `nn.MultiheadAttention`, `torch.autograd.grad`, real forward hooks. No GPU needed at this scale (~160k params); Colab just gives a clean torch install your GPU-less laptop can't. |
| `src/vit_model.py` + `src/interpretability.py` | **Here, already run** | A from-scratch, gradient-checked reverse-mode autograd engine (`src/micrograd_tensor.py`) implementing exactly the ops a ViT needs (matmul, softmax, LayerNorm, GELU, cross-entropy) — because this sandbox has no internet to install torch. Verified against finite-difference gradients (max abs error 4×10⁻¹¹) before being trusted for anything. |

I will not claim a PyTorch run I can't perform here. Every number and plot
below came from a real training run and real backward passes in NumPy; the
Colab notebook is the identical architecture and identical three
interpretability techniques in real PyTorch, ready to run.

## What's standard/published vs. what's the novel combination

- **Standard, published, correctly attributed:** the ViT architecture
  (Dosovitskiy et al., 2021), Logit Lens (nostalgebraist, 2020 — originally
  for decoder-only language models), and Attention Rollout (Abnar &
  Zuidema, 2020) are all established techniques used here exactly as
  published, applied to their intended target (a transformer's residual
  stream and attention matrices).
- **The novel part:** applying Logit Lens — a technique built for language
  models — to a Vision Transformer's CLS token trajectory over hyperspectral
  input, and combining it with gradient-based *spectral band* attribution
  (rather than the usual *spatial pixel* saliency maps) to get a
  band-resolution interpretability signal. That specific combination, for
  this domain, is the underexplored part — not the individual techniques.

## The model

Tokens are spatial patches (4x4 pixels), each carrying its **full 135-band
spectral vector** as the feature dimension before projection — i.e. spatial
self-attention over tokens that each encode a complete spectral signature,
matching the brief's framing of the bands as "a massive feature space."
3 transformer encoder blocks, 4 attention heads, embed_dim=48 (~160k
parameters total — deliberately small; this is a demonstration/validation
model, not a foundation model).

Trained on simulated 16x16-pixel scenes across 4 classes (`healthy_field`,
`early_blight_patch`, `water_stress_gradient`, `mixed_landuse`), reusing the
physically-documented absorption-feature simulator from the earlier Sparse
Autoencoder project (chlorophyll, red-edge, leaf-water bands) but now
arranged into spatially coherent fields with a disease patch that spreads
from one corner, a water-stress irrigation gradient, and a land-use
patchwork — see `src/hyperspectral_cube_simulator.py`. Reaches 100%
validation accuracy on held-out scenes within ~10 epochs (`outputs/`
training log in the run history).

## Results actually produced (NumPy-autograd run, this sandbox)

**1. Logit Lens: the model commits to its answer almost immediately.**
On a held-out `early_blight_patch` scene, class probability for the correct
class is already 87.7% after layer 0, hits 99.9%+ by layer 1, and stays
there — the deeper layers aren't changing the decision, just sharpening it.
That's a genuine finding about *this* small model's depth-efficiency, not
assumed going in.

**2. Attention Rollout correctly localizes the disease epicenter — without
ever being told where it is.** The simulated scene places peak disease
severity at the top-left spatial corner. Attention rollout (aggregating
attention across all 3 layers with residual correction, per Abnar & Zuidema)
assigns that same top-left patch the highest relevance (0.168 in the single
demo scene). Checked across **20 independent held-out scenes**: mean
relevance on the true epicenter patch is **0.187 ± 0.111**, vs. a uniform/
chance baseline of 0.063 (1/16 patches) — a consistent, non-circular result,
not a cherry-picked single case.

**3. Spectral band attribution independently recovers the correct physical
mechanism.** Averaged over the same 20 held-out scenes, the top attributed
bands for the disease-detection decision are:

| Wavelength | Mean attribution | Physical match |
|---|---|---|
| 465nm | 0.0115 ± 0.0035 | Chlorophyll-b / carotenoid absorption region |
| 495nm | 0.0105 ± 0.0033 | Chlorophyll-b / carotenoid absorption region |
| 665nm | 0.0089 ± 0.0028 | Adjacent to chlorophyll-a absorption (680nm) |
| 705nm | 0.0022 ± 0.0003 | Red-edge inflection region (~712nm) |

Nobody told the model that disease was simulated via chlorophyll
degradation and a red-edge shift — it discovered, purely from pixel data and
labels, to attend to exactly the bands a plant physiologist would check.
See `outputs/01_interpretability_dashboard.png`.

## Honest limitations

1. All data is simulated from documented absorption physics, not a real
   Pixxel Firefly cube — the next real step is running this exact pipeline
   (unchanged) on real L2A data.
2. The model is small (~160k params, 3 layers) and the task is a clean
   4-class scene classification — a production system would need many more
   classes, real spatial resolution, and almost certainly a deeper network,
   which would make Logit Lens curves and rollout patterns considerably more
   interesting (and might *not* show single-layer commitment the way this
   toy model does).
3. Attention rollout is a known-imperfect approximation (it assumes
   attention distributes relevance additively/linearly through the network,
   which real transformers don't always respect) — treat it as a useful
   diagnostic, not ground truth about what the model "really" uses.
4. The 20-scene consistency check is real but still small-sample; a
   production validation would want hundreds of scenes and formal
   statistical testing.

## File map

```
vit_hyperspectral_interp/
├── README.md
├── colab/
│   └── hyperspectral_vit_colab.ipynb      <- real PyTorch ViT + hooks, run here
├── src/
│   ├── micrograd_tensor.py                 <- gradient-checked autograd engine
│   ├── hyperspectral_cube_simulator.py     <- spatial hyperspectral scene simulator
│   ├── vit_model.py                        <- ViT built on micrograd_tensor
│   └── interpretability.py                 <- logit lens, attention rollout, band attribution
└── outputs/
    ├── 01_interpretability_dashboard.png
    └── trained_model.pkl
```
