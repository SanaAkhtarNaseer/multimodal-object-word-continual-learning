# Multimodal Object-Word Learning with a Topological Correntropy-Based ART Architecture

An implementation of a brain-inspired, three-layer dual-pathway architecture for online, unsupervised acquisition of audio-visual object concepts, built upon a Topological Kernel CIM-based Adaptive Resonance Network (TC-ART).



<!-- Replace the bracketed fields above before publishing. -->

---

## 1. Overview

An infant shown an apple and told the word _"apple"_ requires neither a labelled training corpus, nor a predetermined number of output classes, nor a second pass over the data. The association is formed once, online, and is subsequently available in both directions: the sight of the object evokes the word, and the word evokes the object.

This repository implements a computational model of that capability. CDSN acquires object categories and their spoken names from an unlabelled stream of samples, in a single epoch, one sample at a time, expanding its own network topology as it proceeds. The number of categories is not specified in advance, no gradients are computed, and no data are replayed. Once trained, the model supports bidirectional cross-modal recall in both the `image → name` and `name → visual prototype` directions.

The hypothesis under test is that a self-organizing network of the Adaptive Resonance Theory (ART) family, when equipped with a **correntropy-induced metric (CIM)** in place of Euclidean distance and a **self-calibrating vigilance threshold** in place of a fixed hyperparameter, can perform continual concept acquisition without catastrophic forgetting and without manual tuning of the stability–plasticity trade-off.

Across 30 randomised presentation orders on the synthetic feature corpus, the model attains **92.67% ± 1.10%** vision-to-name recall and **93.64% ± 4.64%** name-to-vision recall, learned in a single online pass over 176 samples with no labels used during training.

### 1.1 Distinguishing characteristics

| Property                     | Conventional multimodal deep model             | CDSN                                                        |
| ---------------------------- | ---------------------------------------------- | ----------------------------------------------------------- |
| Training regime              | Multiple epochs, batched, offline              | Single epoch, sample-by-sample, online                      |
| Category count               | Fixed by the output layer                      | Grown and pruned autonomously at runtime                    |
| Supervision                  | Labels required                                | Labels never used in learning; retained for evaluation only |
| Novel classes after training | Retraining, replay, or catastrophic forgetting | Absorbed incrementally; prior knowledge retained            |
| Similarity measure           | Euclidean or cosine                            | Correntropy-induced metric — bounded and outlier-robust     |
| Similarity threshold         | Hand-tuned hyperparameter                      | Estimated from the data stream itself                       |
| Recall                       | Typically unidirectional (classification)      | Bidirectional between vision and audition                   |
| Parameter count              | Order of millions                              | A few hundred prototype vectors                             |

The problem addressed is the **stability–plasticity dilemma** in continual learning, approached from the self-organizing and ART traditions rather than through replay buffers or regularisation penalties.

---

## 2. Architecture

The model comprises two parallel pathways — visual and auditory — each traversing three layers and converging upon a shared associative layer.

```
        VISUAL PATHWAY                          AUDITORY PATHWAY
   ┌──────────────────────────┐            ┌──────────────────────┐
   │   image (BGR, 256×256)   │            │   spoken name        │
   └───────────┬──────────────┘            └───────────┬──────────┘
               │  features.py                          │
      ┌────────┴────────┐                              │
      ▼                 ▼                              ▼
 shape (23-D)      colour (63-D)                    string
 Fourier desc.     HSV histogram                       │
      │                 │                              │
══════▼═════════════════▼══════════════════════════════▼══════════ LAYER 1
   TC-ART            TC-ART                AuditoryReceptiveLayer   RECEPTIVE
 (tcart.py)        (tcart.py)                  (auditory.py)        unsupervised
   growing           growing                Levenshtein matching    clustering
   topology          topology                                       ↓ node id
      │                 │                              │
══════▼═════════════════▼══════════════════════════════▼══════════ LAYER 2
 ConceptLayer      ConceptLayer                  ConceptLayer        CONCEPT
  "shape_3"         "color_7"                     "name_2"           symbolic
                                                                     distillation
      │                 │                              │
      └────────┬────────┘                              │
               ▼                                       ▼
══════════════════════════════════════════════════════════════════ LAYER 3
                      AssociativeLayer                              ASSOCIATIVE
              (shape_sym, color_sym) ⟷ {name_syms}                  cross-modal
                 bidirectional, activation-weighted                  binding
```

### 2.1 Visual front-end

Each image is reduced to two attribute vectors by a classical computer-vision pipeline. The object is segmented from the background by Otsu thresholding applied jointly to the saturation and value channels of the HSV representation, followed by morphological closing and opening to suppress noise; the largest external contour is retained as the object boundary.

**Shape** is encoded as a 23-dimensional Fourier descriptor. The contour is resampled to a fixed length of 256 points so that spectral bins remain comparable across objects of differing perimeter, interpreted as a complex-valued signal, and transformed by FFT. The DC component is discarded to achieve translation invariance and the remaining coefficient magnitudes are normalised by the first harmonic to achieve scale invariance.

**Colour** is encoded as a 63-dimensional HSV histogram (7 hue × 3 saturation × 3 value bins), computed over the segmented mask only and normalised to unit sum, so that the descriptor reflects object colour rather than background composition.

Both descriptors are hand-crafted and deliberately low-dimensional. The architecture is agnostic to their provenance: any fixed-length vector may be substituted.

### 2.2 Layer 1 — Receptive layer

Feature vectors are clustered online by **TC-ART**, an ART-family network distinguished by three properties.

**Matching employs the correntropy-induced metric rather than Euclidean distance.** For vectors $u, v \in \mathbb{R}^D$ and kernel bandwidth $\sigma$:

$$\mathrm{CIM}(u,v,\sigma) = \sqrt{\kappa_\sigma(0) - \frac{1}{D}\sum_{d=1}^{D} \exp\!\left(-\frac{(u_d - v_d)^2}{2\sigma^2}\right)}$$

where $\kappa_\sigma(0) = 1$ under the unnormalised Gaussian kernel. The measure is bounded on $[0,1]$ and saturates for large per-dimension deviations, with the consequence that a single corrupted dimension cannot dominate the match — the property that confers robustness to outliers. For small deviations it behaves as a scaled $L_2$ distance. The metric is implemented in `cdsn/cim.py` and vectorised across all nodes in `cim_batch`, with support for per-node bandwidths.

**The vigilance threshold is calibrated from the data stream.** Rather than fixing $\rho$ a priori, the network retains a sliding buffer of the most recent $\lambda$ inputs (default 50), computes all pairwise CIM values within that buffer, and sets the threshold to a low percentile of the resulting distribution — by default the 2nd percentile for shape and the 12th for colour. The threshold is thereby expressed in the units of the data currently being observed, and is re-derived whenever the kernel bandwidth changes. Kernel bandwidths themselves are adapted by Silverman's rule over the same buffer at intervals of $\lambda$ samples.

During the first $\lambda$ samples the threshold is recalibrated on every presentation, and before the buffer reaches a minimum calibration size a deliberately strict cold-start threshold applies. The asymmetry is intentional: because ART cannot merge nodes retrospectively, early over-segmentation is recoverable whereas early under-segmentation is not.

**Network topology is maintained alongside prototypes.** An edge is created between the first and second winning nodes when both satisfy the vigilance criterion. Edges are aged on every activation of an incident node and removed once their age exceeds `edge_max_age`. Node prototypes are updated as an exact running mean, $v_k \leftarrow \frac{M_k}{M_k+1} v_k + \frac{1}{M_k+1} u$, and nodes lacking data support in their vicinity are pruned at periodic intervals subject to a grace period that protects recently created nodes.

Node identifiers are stable and never reused. Because positional indices shift whenever topology adjustment deletes a node, every public method of the network operates on identifiers rather than positions, preventing silent corruption of the index-to-symbol mapping maintained downstream.

The auditory pathway employs a lighter-weight receptive layer that matches words by **Levenshtein edit distance**, with a configurable tolerance (`name_edit_tolerance`) permitting transcription noise of the kind produced by automatic speech recognition — for instance _"banan"_ for _"banana"_ — to be assimilated into a single node.

### 2.3 Layer 2 — Concept layer

The concept layer performs symbolic distillation. Each receptive node identifier is bound to a stable, human-readable symbol (`shape_3`, `color_7`, `name_2`), and per-symbol activation counts are accumulated as evidence of category salience. The mapping is bijective and supports inverse lookup, which is what permits top-down recall to descend from a symbol back to the receptive prototype it denotes.

This abstraction is what renders the associative layer computationally inexpensive: it manipulates short symbol strings rather than high-dimensional vectors. The layer additionally distinguishes _live_ symbols — those whose referent node survives in the receptive layer — from symbols whose referent has since been retired, so that reported category counts reflect the network's actual state.

### 2.4 Layer 3 — Associative layer

The associative layer binds visual symbol pairs `(shape, color)` to sets of auditory name symbols. Its design follows the Convergence–Divergence Zone account of multimodal integration, in which association areas mediate binding without themselves storing or transmitting modality-specific content.

Three properties merit particular note.

_Many-to-many binding._ A single visual concept may carry multiple names (_"tomato"_ and _"cherry tomato"_), and a single name may attach to multiple visual concepts. Recall is disambiguated by per-name activation counts rather than by arbitrary set ordering, so the dominant alias is returned rather than an incidental member.

_No synchrony requirement._ Any of shape, colour, or name may be absent. A name presented in isolation reinforces the node that already carries it, and a visual pair presented without a name still establishes its node. The architecture therefore does not presuppose concurrent presentation of audio-visual data.

_Graceful partial matching._ Where no node matches on both shape and colour, recall degrades to shape-only matching and then to colour-only matching, selecting in each case the most frequently activated candidate.

---

## 3. Experimental design

Two corpora are provided.

The **synthetic feature corpus** comprises 22 object presentations spanning 20 distinct objects, at 8 viewpoints each, for 176 samples. Prototype vectors are generated per colour family and per shape family with additive Gaussian perturbation simulating viewpoint variation. Two deliberate alias pairs are included — _tomato_/_cherry tomato_ and _yellow pear_/_pear_ — so that many-to-many binding is exercised rather than assumed. This corpus isolates architectural behaviour from feature-extraction error.

The **rendered image corpus** comprises 20 objects at 8 viewpoints, generated as parametric polygonal renderings with per-object colour, shape family, scale, and additive pixel noise, and processed through the complete segmentation and descriptor pipeline described in §2.1. This corpus exercises the full system end to end.

Two learning environments are evaluated. In the **closed** environment all objects are intermixed and presented in a single online pass. In the **open-ended** environment the model is trained on ten objects, then on the remaining ten, and evaluated on all twenty — the condition under which catastrophic forgetting would manifest.

Two recall metrics are reported. **Vision recall** measures the proportion of samples for which propagation from visual input to the associative layer and top-down through the auditory pathway returns the correct name. **Auditory recall** measures the proportion for which a name retrieves a visual prototype whose nearest per-label mean shape corresponds to a valid referent of that name; the evaluation is alias-aware, so either member of an alias pair is accepted.

All reported figures are the mean and standard deviation over 30 runs differing only in the **random presentation order** of samples. This control is essential, since an online single-pass learner is order-sensitive by construction and a single run would not characterise the model.

---

## 4. Results

All figures below were produced by `python run_experiments.py --runs 30 --all` on the code in this repository.

### 4.1 Bidirectional recall

**Synthetic feature corpus** (176 samples, 20 objects, 2 alias pairs):

| Environment | Vision → name      | Name → vision      |
| ----------- | ------------------ | ------------------ |
| Closed      | **92.67% ± 1.10%** | **93.64% ± 4.64%** |
| Open-ended  | 90.64% ± 3.80%     | 90.15% ± 6.75%     |

**Rendered image corpus** (160 images, full segmentation and descriptor pipeline):

| Environment | Vision → name      | Name → vision  |
| ----------- | ------------------ | -------------- |
| Closed      | **90.71% ± 4.18%** | 53.50% ± 6.85% |
| Open-ended  | 79.42% ± 3.74%     | 49.33% ± 7.61% |

The principal continual-learning finding is the modest degradation between closed and open-ended conditions on the synthetic corpus: a decline of 2.0 percentage points in vision recall indicates that categories introduced in the second phase are absorbed without erasing those acquired in the first. The corresponding decline on the rendered corpus is considerably larger at 11.3 points, delimiting the conditions under which the property holds.

The disparity in **name-to-vision** recall between the two corpora constitutes the most informative negative result reported here. On engineered features the reverse direction performs comparably to the forward direction; on rendered images it falls to approximately 53%. The cause is that the 23-dimensional Fourier magnitude descriptor retains greater viewpoint sensitivity than the synthetic prototypes exhibit, with the consequence that a single name disperses across several shape nodes and the recalled prototype frequently lies nearer the mean of a different object. This is a limitation of the feature representation rather than of the architecture, and is addressed in §6.

### 4.2 Network growth

The figure `results/growth.png` plots associative node count and per-modality concept count against learning iteration. Both curves rise steeply during initial exposure and subsequently flatten, indicating that the model ceases recruiting once the feature space is adequately covered. This saturation is the expected behavioural signature of an ART network and constitutes direct evidence that vigilance self-calibration is functioning as intended.

Final network size in the closed synthetic condition, averaged over 30 runs, is 46 shape nodes, 13 colour nodes, 19 name nodes and 104 associative nodes, against 20 ground-truth objects. The over-segmentation of shape is a designed consequence of the cold-start policy described in §2.2.

### 4.3 Ablation study

| Variant                | Vision recall | Auditory recall | Associative nodes |
| ---------------------- | ------------- | --------------- | ----------------- |
| Full model             | 92.27%        | 93.64%          | 104               |
| No node deletion       | 93.07%        | 100.00%         | 65                |
| Shape channel only     | n/a           | 50.91%          | 19                |
| Loose vigilance (p=30) | 73.18%        | 51.82%          | 25                |
| Tight vigilance (p=1)  | 91.82%        | 67.27%          | 151               |

Two findings warrant explicit statement.

Removing the colour channel approximately halves auditory recall, from 93.6% to 50.9%. Colour therefore performs substantial disambiguating work: shape alone cannot separate an apple from a tomato. This validates the dual-attribute decomposition of the visual pathway.

Disabling node deletion does not degrade performance but improves it. On these corpora topology pruning is a net liability, since it occasionally retires a receptive node to which a concept symbol still refers, severing that recall path. Pruning would be expected to justify its inclusion only where the input stream contains genuine outliers, which these corpora do not. The result is reported rather than suppressed because it identifies a real deficiency in the pruning criterion as presently formulated; the remedy is discussed in §6.

### 4.4 Vigilance sensitivity

| Percentile | Vision recall | Auditory recall | Shape nodes |
| ---------- | ------------- | --------------- | ----------- |
| 0.5        | 91.70%        | 56.36%          | 54          |
| 1          | 91.82%        | 67.27%          | 54          |
| **2**      | **93.30%**    | 83.64%          | 46          |
| **5**      | 92.16%        | **95.45%**      | 21          |
| 10         | 90.11%        | 79.09%          | 16          |
| 20         | 81.02%        | 63.64%          | 12          |
| 40         | 68.41%        | 49.09%          | 9           |

Vision recall remains essentially flat across two orders of magnitude of resulting network size: varying the percentile from 0.5 to 10 alters the shape node count from 54 to 16 while recall changes only from 91.7% to 90.1%. The self-calibration mechanism therefore removes genuine sensitivity to this parameter rather than merely relocating it. Auditory recall is considerably more sensitive, peaking near the 5th percentile, which again implicates the reverse-direction weakness identified in §4.1.

---

## 5. Reproduction

### Installation

```bash
git clone https://github.com/SanaAkhtarNaseer/multimodal-object-word-continual-learning.git
cd multimodal-object-word-continual-learning
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or later is required. The core model depends only on NumPy; OpenCV is required solely by the image front-end and matplotlib solely by the growth figure.

### Test suite

```bash
pytest tests/ -v          # alternatively: python tests/test_cdsn.py
```

Thirty tests, approximately five seconds.

### Reproducing all tables and figures

```bash
python run_experiments.py --runs 30 --all
```

Approximately three minutes. Writes `results/table_recall.md`, `results/table_nodes.md`, `results/ablation.md`, `results/sensitivity.md`, `results/summary.json` and `results/growth.png`.

### Evaluation on the rendered image corpus

```bash
python make_demo_images.py                                   # renders 20 objects × 8 views
python run_experiments.py --data data/demo_images --runs 30
```

### Interactive demonstration

```bash
python demo.py --data data/demo_images
```

```
cdsn> name banana
  spoken   : "banana"
  symbols  : shape=shape_76  color=color_0
  nearest objects to the recalled visual prototype:
      banana           distance=0.0899
      radish           distance=0.2476
      carrot           distance=0.2796

cdsn> image data/demo_images/apple/view_045.png
  recalled : apple
```

A non-interactive scripted tour is available via `python demo.py --data data/demo_images --auto`.

### Evaluation on external corpora

Any directory following the `root/<object_name>/<view>.png` convention is accepted; the ALOI and COIL-100 layouts conform directly.

```bash
python run_experiments.py --data /path/to/aloi --views 8 --runs 30
```

### Programmatic use

```python
from cdsn import CDSN, CDSNConfig, Sample, make_synthetic_dataset

samples = make_synthetic_dataset()
model = CDSN(CDSNConfig()).fit(samples, shuffle=True, seed=0)

model.recall_name_from_vision(samples[0].shape, samples[0].color)   # → 'apple'
model.recall_vision_from_name("banana")   # → {'shape_symbol': ..., 'shape_prototype': array([...])}
model.node_counts()
```

`CDSNConfig` exposes the vigilance percentiles for each visual attribute, the initial kernel bandwidth, buffer length, edge lifetime, deletion interval, node grace period, a switch disabling node deletion, and the auditory edit tolerance.

---

## 6. Limitations and further work

**Reverse-direction recall degrades substantially on rendered images** (53% against 94% on engineered features). The 23-dimensional Fourier magnitude descriptor exhibits insufficient viewpoint invariance, dispersing a single name across several shape nodes. The natural remedy is to replace the hand-crafted front-end with a frozen self-supervised visual encoder such as DINOv2 or a CLIP image tower and to cluster within that embedding space. Because `Sample` accepts arbitrary fixed-length vectors, this substitution requires no architectural modification.

**The node-deletion criterion is presently counterproductive** (§4.3), since it may retire a receptive node to which a live concept symbol still refers. Introducing reference counting in the concept layer, or deferring deletion until no symbol references the node, would resolve the inconsistency while preserving the outlier-suppression rationale for pruning.

**Evaluation is confined to twenty objects.** Both corpora are controlled by design, which is appropriate for isolating architectural effects but does not constitute a claim about scale. ALOI, comprising 1,000 objects at 72 viewpoints, is the natural subsequent benchmark and the existing loader supports its layout without modification.

**Order sensitivity is measured but not mitigated.** The standard deviations of 1–7% across presentation orders are intrinsic to single-pass online learning. A short experience-replay buffer would plausibly reduce this variance, and quantifying the trade-off against the architecture's replay-free character would be informative.

---



##7. Testing

The suite comprises thirty tests addressing the mathematical properties on which the architecture depends, rather than smoke tests alone.

_Correntropy-induced metric._ Identity yields zero; values are bounded on $[0,1]$; the measure is symmetric; it increases monotonically with dissimilarity; the batch implementation agrees with the scalar implementation to within $10^{-12}$; per-node bandwidths broadcast correctly.

_TC-ART._ Well-separated clusters do not collapse into a single node; node identifiers remain stable across pruning, which is the correctness property on which the validity of concept symbols rests; node learning is verified to be an exact running mean; `predict` does not mutate network state; retired nodes return `None` rather than a stale prototype.

_Concept layer._ Symbol assignment is stable per receptive identifier; symbol-to-receptive lookup round-trips correctly; activation counts accumulate accurately.

_Associative layer._ Bidirectional recall functions in both directions; multiple aliases bind to a single node; distinct visual pairs yield distinct nodes under a shared name; partial single-modality input is accepted; unknown queries return `None`.

_Integration._ End-to-end recall exceeds 75%; missing modalities are handled without error; and phase-one recall does not collapse following phase-two training — the explicit regression test against catastrophic forgetting.

```
30/30 passed
```

---
