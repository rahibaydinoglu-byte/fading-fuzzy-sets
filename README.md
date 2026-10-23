# Fading fuzzy sets — code and results

Code accompanying the article
**"Fading fuzzy sets: an axiomatic characterization of membership aging and its consequences for temporal rank stability"**
by Rahib Imamguluyev (Department of Information Technologies, Baku Business University, Baku, Azerbaijan; rahib.imamguluyev@bbu.edu.az), submitted to *Fuzzy Optimization and Decision Making*.

The scripts reproduce every number, table and figure in the article. All random seeds are fixed (20261023).

## Contents

| File | Reproduces |
|---|---|
| `fading_supplement.py` | Table 3, crossing times in Section 10, sharpness of Theorems 4 and 5, random checks (Appendix A) |
| `theory_checks.py` | exact zero counts, attainment of Corollary 3 for n = 3, frequency of rank exchanges (Remark 6) |
| `experiment.py` | empirical study of Section 11: model fits, Table 4, Friedman test, ablation (Table 7), sensitivity (Fig. 3) |
| `post_analysis.py` | Table 5 (Wilcoxon tests with Holm correction), Table 6 (strata), bootstrap AUC intervals, half-lives |
| `figs_fodm.py` | Figures 1–3 (EPS, PDF, PNG) |
| `*.json` | the outputs used in the article (scripts write them to the repository folder) |

## Data

The empirical study uses the public Bitcoin OTC and Bitcoin Alpha trust networks of the Stanford Network Analysis Project
(https://snap.stanford.edu/data/soc-sign-bitcoinotc.html, https://snap.stanford.edu/data/soc-sign-bitcoinalpha.html;
Kumar et al., 2016). The data are not redistributed here. Download them with

```
python download_data.py
```

The files used in the article have SHA-256 checksums
`76bd9d8f1d3ff9a1813d9fc8e6902a0ee4d0a2f8c1003842dbc9ec79149ab60c` (OTC) and
`1b2a970f327d0ceba0c57bd5919670257cbe4cc0704e2ddac09abc4b08e2ca4d` (Alpha).

## Running

```
pip install -r requirements.txt
python download_data.py
python fading_supplement.py
python theory_checks.py
python experiment.py        # about 30-60 minutes
python post_analysis.py
python figs_fodm.py
```

## Citation

If you use this code, please cite the article (full reference to be added after publication).

## License

MIT License (see `LICENSE`).
