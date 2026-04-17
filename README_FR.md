# Sieve Color Space (SCS)

[English](README.md)

Un substrat géométrique pour la couleur dérivé d'une seule hypothèse (`s = 1/2`) sur un système dynamique de gaps entre nombres premiers — le cadre de la [Théorie de la Persistance](https://zenodo.org/records/19583187). SCS est proposé comme une *couche* géométrique principielle dans une architecture d'apparence des couleurs factorisée, **aux côtés** des outils existants (CIECAM02, CIEDE2000, iCAM, CAM02-UCS, J<sub>z</sub>A<sub>z</sub>B<sub>z</sub>), et non comme un remplacement.

**Ce qui est dérivé sans paramètre ajusté :** le simplexe à trois canaux, l'ordre `γ₃ > γ₅ > γ₇` qui coïncide avec l'ordre des bandes passantes des cônes L>M>S, la métrique de Fisher sur le simplexe, les géodésiques de Bhattacharyya et Fisher–Bernoulli, et la règle de somme saturation–luminance.

**Ce qui reste ajusté / mesuré :** les métriques hybrides de différence de couleur (SCS+CIECAM02 et ΔE_SCS00) utilisent des poids régressés par Ridge sur la géométrie dérivée. L'article les étiquette systématiquement comme des hybrides. L'adaptation chromatique, la sensibilité spectrale au-delà de la linéarisation HPE, les conditions d'observation et le flare ne sont **pas** modélisés par SCS (voir *Portée et limites* dans l'article) ; CIECAM02 et CIEDE2000 restent les outils appropriés là où ces effets dominent.

[**Démonstration en ligne**](https://igrekess.github.io/SieveColorSpace/demonstration/demo.html)
· [PDF anglais](PT_COLOR.pdf)
· [PDF français](PT_COLOR_FR.pdf)
· [**Leaderboard**](leaderboard.md) — benchmark ouvert, pull requests bienvenues
· [Résultats V4 fMRI](docs/v4_results.md) — validation biologique

---

## Reproduire les chiffres du papier

Chaque affirmation du papier est reproductible depuis un script dans `scripts/`. Clone propre + `pip install -e .` + ~1 min de CPU donnent les quatre nombres clés :

```bash
# 1. Orientation des ellipses MacAdam — SCS gagne 18/25, RMS Δθ = 37,8°
python3 scripts/macadam_test.py

# 2. Différence de couleur SCS pure sur COMBVD — r ≈ 0,500 (zéro poids ajusté)
python3 scripts/delta_e_scs.py

# 3. Hybride SCS + CIECAM02 sur COMBVD — r = 0,824 (6 poids ajustés)
python3 scripts/scs_cam02_hybrid.py

# 4. ΔE_SCS00 vs CIEDE2000 sur COMBVD — r = 0,893 vs 0,878, p < 0,0001
python3 scripts/delta_e_scs00.py

# 5. Poids des canaux V4 fMRI — L−M = 0,37 vs prédiction SCS γ₃/Σγ = 0,385
python3 scripts/v4_summary.py      # reproduction rapide depuis la CSV pré-calculée
python3 scripts/v4_refined_analysis.py  # pipeline complet (nécessite OpenNeuro ds005521)
```

Chaque script imprime son résultat à côté de la valeur citée dans le papier. Si votre exécution donne un résultat différent, ouvrez une issue — nous voulons le savoir.

---

## Performance honnête sur COMBVD (3813 paires, validation croisée 5-fold)

| Méthode | Poids ajustés | r | vs CIEDE2000 |
|---------|--------------:|--:|:--|
| SCS pur (feature unique `delta_e_scs`) | **0** | **0,492** | base géométrique |
| CIELAB | 3 | 0,755 | –14 % |
| SCS pur + Ridge (d_lum, d_chrom) | 2 | 0,642 | –27 % |
| Hybride SCS + CIECAM02 | 6 | 0,824 | –6 % |
| **CIEDE2000** | 5 | 0,878 | référence |
| **ΔE_SCS00** (CIEDE2000 + Fisher–Bernoulli) | 5 | **0,893** | **+1,8 %**, p < 0,0001 |

La métrique SCS pure (zéro paramètre ajusté) est **en dessous** de CIELAB globalement. Elle gagne dans deux régimes spécifiques :
- **Région sombre** (L\* < 25) : r = 0,625 vs 0,558 pour CIELAB
- **Orientation des ellipses MacAdam** : 18/25 victoires, RMS Δθ = 37,8° vs 52,0° pour CIELAB

La valeur de SCS comme métrique opérationnelle est celle d'un **canal d'information additif** au-dessus des modèles corticaux (CIECAM02, CIEDE2000) — jamais comme remplacement.

La formule `ΔE_SCS00` :
```
ΔE_SCS00 = w₀ + w₁·ΔE₀₀ + w₂·d_lum + w₃·ΔE₀₀² + w₄·ΔE₀₀·d_lum + w₅·d_lum²
```
où `d_lum = 2 |arcsin(√ℓ₁) − arcsin(√ℓ₂)|` est dérivé de `s = 1/2` sans paramètre ajusté. Les six poids `w₀…w₅` sont régressés par Ridge sur COMBVD (`α = 1`, 5-fold CV, graine 42). Le terme clé est l'interaction `ΔE₀₀ · d_lum` — de l'information que le modèle cortical seul ne porte pas.

---

## Installation

```bash
pip install -e .
```

Ou depuis PyPI :
```bash
pip install scs
```

Dépendances requises : `numpy`, `scipy`, `pandas`, `scikit-learn`, `colour-science` (pour CIECAM02 dans l'hybride). `matplotlib` pour les figures.

---

## Démarrage rapide

```python
from scs import delta_e, to_scs, fisher_luminance, gft_check

# Différence de couleur (géodésique pure, 0 paramètre)
d = delta_e([0.95, 1.0, 1.09], [0.60, 0.50, 0.30])
print(f"ΔE_SCS = {d:.4f}")

# XYZ → coordonnées SCS
c = to_scs([0.95, 1.0, 1.09])
print(f"ℓ={c.ell:.2f}  S={c.S:.3f} bits  θ={c.hue:.0f}°")
print(f"π = ({c.pi[0]:.3f}, {c.pi[1]:.3f}, {c.pi[2]:.3f})")

# Géodésique de luminance Fisher–Bernoulli (utilisée dans ΔE_SCS00)
d_lum = fisher_luminance(Y1=50, Y2=45)

# Depuis des valeurs CIELAB
from scs import delta_e_lab
d = delta_e_lab(50, 25, 10, 52, 28, 8)

# Règle de somme S + L (identité générique sur toute distribution à 3 issues)
S, L, total, err = gft_check([0.4, 0.35, 0.25])
print(f"S + L = {total:.6f} (err = {err:.1e})")
```

### Résumé API

| Fonction | Rôle |
|----------|------|
| `delta_e(xyz1, xyz2)` | Différence de couleur SCS (géodésique pure, 0 paramètre) |
| `delta_e_lab(L1,a1,b1, L2,a2,b2)` | Idem depuis CIELAB |
| `to_scs(xyz)` | XYZ → `SCSColor(ell, S, hue, pi)` |
| `fisher_luminance(Y1, Y2)` | Géodésique Fisher–Bernoulli `d_lum` |
| `saturation(pi)` | Divergence de Kullback–Leibler par rapport à uniforme |
| `luminance_entropy(pi)` | Entropie de Shannon `H(π)` |
| `gft_check(pi)` | Vérifier `S + L = log 3` |

### Constantes (toutes dérivées de `s = 1/2` à `μ* = 15`)

| Constante | Valeur | Source |
|-----------|--------|--------|
| `MU_STAR` | 15 | Point fixe unique (Théorème T5) |
| `GAMMAS` | (0,808 ; 0,696 ; 0,595) | Dimensions anomales à `μ* = 15` |
| `PRIMES` | (3, 5, 7) | Nombres premiers actifs |
| `Q_REL` | 13/15 | Branche sommet de la bifurcation |
| `Q_THERM` | e^(−1/15) | Branche arête |
| `W_LUM` | 3/4 | Poids luminance `N/(N+1)` |
| `W_CHROM` | 1/4 | Poids chromaticité `1/(N+1)` |

---

## Scripts

Tous les scripts sont dans `scripts/`. Ils résolvent les chemins de données depuis la racine du repo.

| Script | Rôle | Claim du papier reproduite |
|--------|------|----------------------------|
| `scs.py` | Module cœur : coordonnées, métriques, règle de somme, auto-test | — |
| `scs_companion.py` | Quantités PT, vérification numérique, génération de figures | — |
| `delta_e_scs.py` | Différence SCS pure (métrique combinée : Fisher + Fubini–Study + bifurcation) | `r = 0,492` baseline |
| `delta_e_scs00.py` | Hybride ΔE_SCS00 = CIEDE2000 + Fisher–Bernoulli | **r = 0,893 vs 0,878**, p < 0,0001 |
| `scs_cam02_hybrid.py` | Fitter hybride SCS + CIECAM02 (Ridge, 5-fold CV) | **r = 0,824** |
| `macadam_test.py` | Validation ellipses MacAdam (25 points) avec métrique combinée | **18/25 victoires, RMS 37,8°** |
| `v4_summary.py` | Reproduction poids canaux V4 depuis CSV pré-calculée | L−M ≈ 0,37 vs γ₃/Σγ = 0,385 |
| `v4_neural_extraction.py` | Extraction V4 fMRI complète (requiert OpenNeuro ds005521) | Génère `v4_bold_response.csv` |
| `v4_refined_analysis.py` | Mapping Ridge V4 → canaux SCS | L−M = 0,373 (écart 3,2 %) |
| `v4_hybrid_model.py` | Canaux opposants V4 comme proxy de CAM02 dans l'hybride COMBVD | r = 0,675 (V4-based, en dessous de CAM02 plein) |
| `v4_analysis_plots.py` | Figures V4 | — |
| `scs_ciede2000_analysis.py` | Analyse comparative SCS vs CIEDE2000 | — |
| `exploratory/` | R&D brouillon : `model20_deep_analysis`, `push_beyond_ciede2000`, `pt_matrix` | Non-structurant |

---

## Article

Le papier est **autonome** : l'appendice fournit les dérivations de tous les résultats de Théorie de la Persistance utilisés (T1 transitions interdites, `s = 1/2`, règle de somme, bifurcation sommet–arête, holonomie, dimensions anomales, nombres premiers actifs, point fixe). Un lecteur n'a pas besoin d'adopter la PT comme tout cohérent pour évaluer les claims SCS — voir le paragraphe explicite *« Lire cet article sans s'engager sur le cadre plus large »* dans l'introduction.

- [PDF anglais](PT_COLOR.pdf)
- [PDF français](PT_COLOR_FR.pdf)

---

## Chaîne de dérivation

```
s = 1/2  →  T₁ transitions interdites  →  T₃ = antidiag(1, 1)
         →  T₅  →  μ* = 15,  premiers actifs {3, 5, 7}
         →  holonomie  →  sin² θ_p,  α_EM ≈ 1/137
         →  Rydberg → Balmer  →  fenêtre visible 380–656 nm
         →  hiérarchie γ_p : γ₃ > γ₅ > γ₇   (coïncide L>M>S)
         →  métrique de Fisher (Čencov, unique à constante près)  →  SCS
```

---

## Problèmes ouverts (collaboration sollicitée)

L'article énonce trois prédictions falsifiables, chacune traitable en trois mois d'expérience. Les esquisses de protocoles et les crochets de préenregistrement sont dans [`leaderboard.md`](leaderboard.md).

- **E1** — Seuil JND nul à la saturation de Koide `S/S_max = 1/√2 ≈ 70,7 %` (2AFC).
- **E2** — Plafond de dimensionnalité chromatique à `3 × 5 × 7 = 105` états (méthode maximum-likelihood sur ensembles métamériques).
- **E3** — Signature de quatrième canal chez les tétrachromates prédite par `γ₁₁`.

Le papier signale aussi explicitement les directions où les auteurs ont besoin d'aide (adaptation chromatique, modélisation de l'observateur individuel, design de protocoles psychophysiques). Correspondance, réplications, contre-expériences et forks sont activement sollicités — ouvrez une issue ou une PR.

---

## Documentation

- [Référence API](docs/api.md) — fonctions, classes, constantes
- [Théorie](docs/theory.md) — fondements mathématiques (condensé)
- [Palettes](docs/colormaps.md) — 12 palettes géodésiques Fisher
- [Format grade](docs/grade-format.md) — format `.scs` pour l'étalonnage colorimétrique
- [Résultats V4 fMRI](docs/v4_results.md) — détails de validation biologique
- [Leaderboard](leaderboard.md) — soumissions benchmark et prédictions ouvertes

---

## Références

Y. Senez, *« The Sieve Color Space: A First-Principles Color Space from the Sieve of Eratosthenes »* (2026). [PDF](PT_COLOR.pdf) · [Zenodo](https://zenodo.org/records/19614967)

Y. Senez, *« Persistence Theory: Mathematical Foundations of Prime Gap Dynamics »*, préprint (2026). [Zenodo](https://zenodo.org/records/19583187)

Littérature adjacente citée dans le papier : Wyszecki & Stiles (2000), Vos & Walraven (1972), Derrington–Krauskopf–Lennie (1984), MacLeod–Boynton (1979), Wuerger et al. (2002), Koenderink (2010), Fairchild (2013), Hofer et al. (2005), Amari (1985), Fairchild & Johnson iCAM (2004), Luo et al. CAM02-UCS (2006), Safdar et al. JzAzBz (2017), Conway et al. (2025, OpenNeuro ds005521).

## Licence

MIT — Yan Senez — [www.dityan.com](https://www.dityan.com)
