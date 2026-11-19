# Thinking pilot on one frontier (exploratory, PREREG R24 E3)

208 level-5 dev problems with all 4 fresh thinking solves; cost in samples of the non-thinking pool; credit in percent; intervals 95% bootstrap over problems. Descriptive; dev only; no test.

## generated rule

| Method (envelope) | Cost | Credit |
| --- | --- | --- |
| sc_1 | 1.00 | 45.2 |
| sc_4 | 4.00 | 54.9 |
| sc_8 | 8.00 | 61.0 |
| sc_12 | 12.00 | 63.5 |
| sc_18 | 18.00 | 64.8 |
| esc_w2_m8 | 4.97 | 60.3 |
| esc_w2_m16 | 7.35 | 62.8 |
| esc_w4_m8 | 6.98 | 61.1 |
| esc_w4_m16 | 12.14 | 64.5 |
| esc_w8_m8 | 8.00 | 61.1 |
| esc_w8_m16 | 14.58 | 64.6 |
| think_1_cap4096 | 4.41 | 24.5 |
| think_1_cap8192 | 6.48 | 56.5 |
| think_1_cap16384 | 8.08 | 77.0 |
| think_1_cap30000 | 8.68 | 83.4 |
| think_vote_2 | 17.36 | 84.1 |
| think_vote_4 | 34.71 | 88.4 |

| Policy | Cost | Credit | Escalated | − envelope [95% CI] |
| --- | --- | --- | --- | --- |
| esc_w2_m2_then_think_cap16384 | 7.06 | 75.0 | 60% | -0.4 [-3.5, +2.3] |
| esc_w2_m2_then_think_cap30000 | 7.59 | 80.7 | 60% | +2.7 [+0.2, +5.0] |
| esc_w2_m4_then_think_cap16384 | 7.21 | 73.6 | 47% | -2.5 [-6.4, +1.0] |
| esc_w2_m4_then_think_cap30000 | 7.69 | 78.8 | 47% | +0.3 [-3.2, +3.5] |
| esc_w2_m8_then_think_cap16384 | 7.93 | 71.4 | 35% | -8.3 [-13.4, -3.6] |
| esc_w2_m8_then_think_cap30000 | 8.34 | 75.8 | 35% | -5.9 [-9.7, -1.5] |
| esc_w2_m8_then_think_read | 6.55 | 66.0 | 35% | -6.8 [-12.4, -1.8] |
| esc_w2_m8_then_nothink_read_untrained | 5.18 | 60.5 |  | -5.5 [-9.9, -1.3] |
| esc_w2_m8_then_nothink_read_step100 | 5.18 | 61.7 |  | -4.4 [-8.8, -0.2] |

Thinking points minus the non-thinking envelope (SC + ESC):

- think_1_cap4096: cost 4.41, -33.7 [-39.4, -28.0]
- think_1_cap8192: cost 6.48, -5.4 [-10.9, -0.2]
- think_1_cap16384: cost 8.08, +14.0 [+8.7, +19.2]
- think_1_cap30000: cost 8.68, +20.2 [+14.6, +25.8]
- think_vote_2: cost 17.36, +19.4 [+13.5, +25.0]
- think_vote_4: cost 34.71, outside the envelope

## prompt_inclusive rule

| Method (envelope) | Cost | Credit |
| --- | --- | --- |
| sc_1 | 1.00 | 45.2 |
| sc_4 | 4.00 | 54.9 |
| sc_8 | 8.00 | 61.0 |
| sc_12 | 12.00 | 63.5 |
| sc_18 | 18.00 | 64.8 |
| esc_w2_m8 | 4.97 | 60.3 |
| esc_w2_m16 | 7.35 | 62.8 |
| esc_w4_m8 | 6.98 | 61.1 |
| esc_w4_m16 | 12.14 | 64.5 |
| esc_w8_m8 | 8.00 | 61.1 |
| esc_w8_m16 | 14.58 | 64.6 |
| think_1_cap4096 | 3.76 | 24.5 |
| think_1_cap8192 | 5.49 | 56.5 |
| think_1_cap16384 | 6.85 | 77.0 |
| think_1_cap30000 | 7.39 | 83.4 |
| think_vote_2 | 14.77 | 84.1 |
| think_vote_4 | 29.54 | 88.4 |

| Policy | Cost | Credit | Escalated | − envelope [95% CI] |
| --- | --- | --- | --- | --- |
| esc_w2_m2_then_think_cap16384 | 6.34 | 75.0 | 60% | -2.2 [-5.4, +0.6] |
| esc_w2_m2_then_think_cap30000 | 6.81 | 80.7 | 60% | +0.7 [-1.9, +3.1] |
| esc_w2_m4_then_think_cap16384 | 6.66 | 73.6 | 47% | -5.5 [-9.6, -1.8] |
| esc_w2_m4_then_think_cap30000 | 7.09 | 78.8 | 47% | -2.9 [-6.1, +0.6] |
| esc_w2_m8_then_think_cap16384 | 7.54 | 71.4 | 35% | -12.0 [-15.4, -7.6] |
| esc_w2_m8_then_think_cap30000 | 7.90 | 75.8 | 35% | -7.7 [-10.6, -4.6] |
| esc_w2_m8_then_think_read | 8.54 | 66.0 | 35% | -17.7 [-22.6, -12.8] |
| esc_w2_m8_then_nothink_read_untrained | 7.36 | 60.5 |  | -22.7 [-28.1, -15.5] |
| esc_w2_m8_then_nothink_read_step100 | 7.37 | 61.7 |  | -21.6 [-26.9, -14.3] |

Thinking points minus the non-thinking envelope (SC + ESC):

- think_1_cap4096: cost 3.76, -31.2 [-36.6, -25.9]
- think_1_cap8192: cost 5.49, -4.4 [-9.6, +0.6]
- think_1_cap16384: cost 6.85, +14.8 [+9.5, +20.0]
- think_1_cap30000: cost 7.39, +20.6 [+15.2, +26.5]
- think_vote_2: cost 14.77, +19.5 [+13.7, +25.2]
- think_vote_4: cost 29.54, outside the envelope

## Thinking read minus fresh thinking solve (sets the read covers)

- read_cap16384_minus_fresh_cap16384: -11.6 [-18.1, -5.2] over 131 problems
- read_cap16384_minus_fresh_cap30000: -21.6 [-28.1, -15.1] over 131 problems

## GPU

- candidate_free: 832 generations, 1.55 h, 7481760 tokens, 1339 tokens/s
- reader_full: 251 generations, 0.54 h, 1308463 tokens, 668 tokens/s
