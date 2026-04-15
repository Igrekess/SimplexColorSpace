# SCS — Sieve Color Space

[English](README.md)

Un espace de couleurs derive de
[la Theorie de la Persistance](https://github.com/yan-senez/persistence-theory),
un framework mathematique fonde sur la dynamique des ecarts entre nombres premiers.
**Entree unique :** `s = 1/2`. **Zero parametres ajustables.**
Un facteur de traduction (m_e = 0,511 MeV) convertit des unites naturelles
du crible vers le SI ; tous les rapports et structures sont derives.


[**Pour une démonstration rapide : **](https://igrekess.github.io/SimplexColorSpace/demonstration/demo.html)[ Cliquer ici](https://igrekess.github.io/SimplexColorSpace/demonstration/demo.html)

**Nouveau (avril 2026) :** La formule SCS00 (CIEDE2000 + geodesique de Fisher-Bernoulli)
**surpasse CIEDE2000** sur COMBVD (r = 0,893 vs 0,878, p < 0,0001).

L'article est desormais autonome avec un appendice mathematique complet.
- [French PDF](article/PT_COLOR_FR.pdf)
- [English PDF](article/PT_COLOR.pdf)


## Installation

```bash
pip install scs
```

Ou depuis ce repo :

```bash
pip install -e .
```

## Package : `scs`

Le package Python fournit les coordonnees couleur, les differences de
couleur et la verification de la loi de conservation — le tout derive
de `s = 1/2` avec zero parametres ajustables.

```python
from scs import delta_e, to_scs, fisher_luminance, gft_check

# --- Difference de couleur (0 parametres) ---
d = delta_e([0.95, 1.0, 1.09], [0.60, 0.50, 0.30])
print(f"DE_SCS = {d:.4f}")

# --- Conversion XYZ → coordonnees SCS ---
c = to_scs([0.95, 1.0, 1.09])
print(f"l={c.ell:.2f}, S={c.S:.3f} bits, h={c.hue:.0f} deg")
print(f"pi = ({c.pi[0]:.3f}, {c.pi[1]:.3f}, {c.pi[2]:.3f})")

# --- Geodesique Fisher-Bernoulli (utilisee dans SCS00) ---
d_lum = fisher_luminance(Y1=50, Y2=45)

# --- Depuis des valeurs CIELAB ---
from scs import delta_e_lab
d = delta_e_lab(50, 25, 10, 52, 28, 8)

# --- Conservation GFT : S + L = log2(3), toujours ---
S, L, total, err = gft_check([0.4, 0.35, 0.25])
print(f"S + L = {total:.6f} (err = {err:.1e})")
```

### Resume de l'API

| Fonction | Role |
|----------|------|
| `delta_e(xyz1, xyz2)` | Difference de couleur SCS (0 parametres) |
| `delta_e_lab(L1,a1,b1, L2,a2,b2)` | Idem, depuis des valeurs CIELAB |
| `to_scs(xyz)` | XYZ → `SCSColor(ell, S, hue, pi)` |
| `fisher_luminance(Y1, Y2)` | Geodesique Fisher-Bernoulli d_lum |
| `saturation(pi)` | D_KL(pi \|\| uniforme) |
| `luminance_entropy(pi)` | H(pi) |
| `gft_check(pi)` | Verifier S + L = log2(3) |

### Constantes (toutes derivees, pas choisies)

| Constante | Valeur | Source |
|-----------|--------|--------|
| `MU_STAR` | 15 | Point fixe unique (T5) |
| `GAMMAS` | (0,808 ; 0,696 ; 0,595) | Dimensions anomales a mu*=15 |
| `PRIMES` | (3, 5, 7) | Premiers actifs |
| `Q_REL` | 13/15 | Branche sommet |
| `Q_THERM` | e^(-1/15) | Branche arete |
| `W_LUM` | 3/4 | Poids luminance N/(N+1) |
| `W_CHROM` | 1/4 | Poids chromaticite 1/(N+1) |

## Performance

| Methode | Parametres | r (COMBVD) | vs CIEDE2000 |
|---------|-----------|------------|--------------|
| SCS pur | **0** | 0,583 | base geometrique |
| CIELAB | 3 | 0,755 | -14% |
| SCS + CAM02 | 6 | 0,824 | -6% |
| **CIEDE2000** | **5** | **0,878** | **reference** |
| **SCS00** | **5** | **0,893** | **+1,8%** |

La formule SCS00 combine le modele cortical de CIEDE2000 avec la
geodesique de luminance Fisher-Bernoulli derivee de la PT :

```
SCS00 = w0 + w1*DE00 + w2*d_lum + w3*DE00^2 + w4*DE00*d_lum + w5*d_lum^2
```

ou `d_lum = 2|arcsin(sqrt(l1)) - arcsin(sqrt(l2))|` a **zero parametres
ajustables** (derive de s = 1/2). Le terme cle est `DE00 * d_lum` :
l'interaction entre la metrique corticale et la geodesique retinienne.

Sur les 25 ellipses de MacAdam, la metrique SCS gagne **18/25**
avec zero parametres contre trois pour CIELAB.

## Article

L'article est **autonome** : un appendice contient les demonstrations
completes de tous les resultats PT (T1, s = 1/2, GFT, bifurcation,
holonomie, dimensions anomales, premiers actifs, point fixe).

- [PDF anglais](article/PT_COLOR.pdf)
- [PDF francais](article/PT_COLOR_FR.pdf)

## Scripts

| Script | Role |
|--------|------|
| `delta_e_scs00.py` | **Formule SCS00** — CIEDE2000 + geodesique Fisher |
| `delta_e_scs.py` | Difference de couleur SCS pure (0 parametres) |
| `scs_companion.py` | Quantites PT, figures, verification numerique |
| `macadam_test.py` | Validation des ellipses de MacAdam |
| `scs_ciede2000_analysis.py` | Analyse comparative SCS vs CIEDE2000 |
| `push_beyond_ciede2000.py` | Exploration systematique de 20+ modeles |
| `model20_deep_analysis.py` | Analyse approfondie du modele SCS00 |

## Chaine de derivation

```
s = 1/2 → T1 → T3 = antidiag(1,1)
        → T5 → mu* = 15, premiers actifs {3,5,7}
        → holonomie → sin^2(theta_p), alpha_EM ~ 1/137
        → Rydberg → Balmer → 380–656 nm (fenetre visible)
        → hierarchie gamma_p → Rouge (p=3) > Vert (p=5) > Bleu (p=7)
        → metrique de Fisher (Cencov, unique) → SCS
```

## Documentation

- [Reference API](docs/api.md) — toutes les fonctions et classes
- [Theorie](docs/theory.md) — fondements mathematiques
- [Palettes](docs/colormaps.md) — les 12 palettes geodesiques de Fisher
- [Format Grade](docs/grade-format.md) — le format de fichier .scs

## References

Y. Senez, *"The Sieve Color Space: A First-Principles Color Space
from the Sieve of Eratosthenes"* (2026).

Y. Senez, *"Persistence Theory: Mathematical Foundations of Prime Gap
Dynamics"*, preprint (2026).

## Licence

MIT — Yan Senez — [www.dityan.com](https://www.dityan.com)
