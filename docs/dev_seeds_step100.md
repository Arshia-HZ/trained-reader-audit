# Dev evaluation of step 100 across reader seeds (R28)

Descriptive, dev only; it selects nothing. Points, means over the 208
level-5 dev problems (H2 and H4a over the problems that have a c1 or c2+
set), SD over problems in brackets. Seed 1's step 100 was chosen by D17 as
the best of three steps, so its value is biased upward; seeds 2 and 3 were
fixed at step 100 by R28.

| Seed | H1a, reader minus matched SC | H2, reader minus its candidate-free solve (c1) | H4a, reader minus vote (c2+) | Read tokens | k* |
| --- | --- | --- | --- | --- | --- |
| reader_s1 | +0.5 (SD 0.144) | +6.4 (SD 0.403, 64 problems) | -2.5 (SD 0.160, 154 problems) | 730 | 8.74 |
| reader_s2 | -0.1 (SD 0.137) | +6.4 (SD 0.401, 64 problems) | -2.5 (SD 0.162, 154 problems) | 553 | 8.59 |
| reader_s3 | +0.1 (SD 0.177) | +8.3 (SD 0.396, 64 problems) | -3.7 (SD 0.191, 154 problems) | 1021 | 9.02 |
| mean over seeds | +0.2 | +7.0 | -2.9 | | |
| SD over seeds (points) | 0.3 | 1.1 | 0.7 | | |
