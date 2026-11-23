# Thinking read against fresh thinking solve, by stratum (exploratory, E4)

Level-5 dev; each read paired with the mean of its problem's 4 fresh solves; means over problems; read minus fresh with a 95% bootstrap over problems.

| Stratum | Reads | Problems | Quantity | Read | Fresh | Read − fresh [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| c0 | 94 | 55 | Credit at 16,384 | 24.5 | 54.1 | -29.5 [-40.5, -19.1] |
| c0 | 94 | 55 | Thinking tokens | 5880 | 13339 | -7459 [-9029, -5861] |
| c0 | 94 | 55 | Epistemic words per 1,000 | 24.72 | 28.76 | -4.04 [-5.59, -2.45] |
| c0 | 94 | 55 | Answer equals a shown one (%) | 51.8 | 6.4 | +45.5 [+33.9, +57.3] |
| c1 | 39 | 33 | Credit at 16,384 | 40.9 | 67.4 | -26.5 [-44.7, -8.3] |
| c1 | 39 | 33 | Thinking tokens | 5056 | 10424 | -5368 [-7221, -3419] |
| c1 | 39 | 33 | Epistemic words per 1,000 | 23.29 | 29.89 | -6.60 [-10.07, -3.29] |
| c1 | 39 | 33 | Answer equals a shown one (%) | 80.3 | 82.6 | -2.3 [-17.4, +13.6] |
| c2+ | 118 | 73 | Credit at 16,384 | 84.9 | 79.5 | +5.5 [-2.1, +12.7] |
| c2+ | 118 | 73 | Thinking tokens | 3074 | 8365 | -5291 [-6280, -4330] |
| c2+ | 118 | 73 | Epistemic words per 1,000 | 22.81 | 30.34 | -7.54 [-9.79, -5.33] |
| c2+ | 118 | 73 | Answer equals a shown one (%) | 94.5 | 89.0 | +5.5 [+0.3, +11.3] |
| all | 251 | 131 | Credit at 16,384 | 57.3 | 68.9 | -11.6 [-17.9, -5.3] |
| all | 251 | 131 | Thinking tokens | 4356 | 10516 | -6160 [-7006, -5304] |
| all | 251 | 131 | Epistemic words per 1,000 | 23.31 | 29.59 | -6.29 [-7.76, -4.90] |
| all | 251 | 131 | Answer equals a shown one (%) | 77.5 | 59.3 | +18.2 [+12.1, +24.6] |

## Headroom

| Policy | Rule | Cost | Credit |
| --- | --- | --- | --- |
| dart | generated | 7.59 | 80.7 |
| gate | generated | 6.94 | 85.4 |
| agree | generated | 8.02 | 84.6 |
| mode | generated | 6.89 | 84.5 |
| dart | prompt_inclusive | 6.81 | 80.7 |
| gate | prompt_inclusive | 6.24 | 85.4 |
| agree | prompt_inclusive | 7.18 | 84.6 |
| mode | prompt_inclusive | 5.92 | 84.5 |

Thinking solves: pass@1 83.4, pass@2 90.1, pass@4 93.8.
