# Results summary

397 candidate queries (308 correct, 89 wrong) from 82 tasks, reviewed by 5 judges. Everything that is tuned is tuned on other tasks (5-fold, grouped by task); intervals are 95% bootstrap over tasks. Labels: lenient.

## Each judge alone

| Judge | Passes correct | Passes wrong | Youden's J | Pattern |
|---|---|---|---|---|
| codegemma:7b | 100% | 99% | 0.01 | rubber stamp |
| codellama:13b | 96% | 82% | 0.14 |  |
| gemma3:12b | 92% | 40% | 0.51 |  |
| gpt-oss:20b | 94% | 15% | 0.80 |  |
| phi4:14b | 96% | 43% | 0.54 |  |

Bug types that most often fool the best single judge (**gpt-oss:20b**), share of those wrong queries it passed:

- natural (n=89): 15%

## Main comparison

| System | Passes correct | Passes wrong | Youden's J | J vs paper recipe |
|---|---|---|---|---|
| Paper recipe: best unanimous jury, picked on training tasks | 89% | 12% | 0.76 [0.67, 0.84] |  |
| Unanimous, all judges | 85% | 7% | 0.78 [0.71, 0.85] | +1.7 [-4.2, +9.8] |
| Reliability-weighted vote, all judges | 88% | 11% | 0.77 [0.68, 0.85] | +0.8 [-0.9, +3.8] |
| Confidence-weighted vote, all judges | 88% | 11% | 0.77 [0.68, 0.85] | +0.5 [-2.7, +3.9] |
| Confidence-weighted, tuned to the recipe's false accept rate | 89% | 13% | 0.75 [0.65, 0.84] | -1.1 [-7.5, +3.7] |
| Dawid-Skene, all judges, no labels at all | 93% | 29% | 0.64 [0.49, 0.77] | -12.6 [-27.4, -0.5] |

Committee the paper recipe picked: codegemma:7b + gpt-oss:20b (2/5 folds), gemma3:12b + gpt-oss:20b + phi4:14b (2/5 folds), gpt-oss:20b + phi4:14b (1/5 folds).
Area under the ROC curve for the full panel: weighted 0.924, confidence 0.964, dawid_skene 0.915.

## The veto effect

| Judge | Correct queries it blocks when added | Wrong queries it blocks when added |
|---|---|---|
| codegemma:7b | 0% | 0% |
| codellama:13b | 3% | 10% |
| gemma3:12b | 6% | 42% |
| gpt-oss:20b | 5% | 76% |
| phi4:14b | 2% | 28% |

No judge blocks more correct queries than wrong ones when added. The least useful is **codegemma:7b**, which blocks only 0% of the wrong queries a jury used to pass.

- Per committee, **weighted** vs unanimous (best-J threshold): Youden's J higher in 8/26 committees, clearly higher in 0, clearly lower in 0.
- Per committee, **confidence** vs unanimous (best-J threshold): Youden's J higher in 21/26 committees, clearly higher in 1, clearly lower in 0.

## Weights learned per SQL-writing model ("adaptive")

| Writer model | Wrong queries | J, pooled weights | J, per-writer weights |
|---|---|---|---|
| codegemma:7b | 18 | 0.75 | 0.71 |
| codellama:13b | 30 | 0.81 | 0.77 |
| gemma3:12b | 17 | 0.86 | 0.72 |
| gpt-oss:20b | 10 | 0.49 | 0.52 |
| phi4:14b | 14 | 0.78 | 0.80 |
| (all) | 89 | 0.77 | 0.74 |

Overall difference (per-writer minus pooled): -3.5 points, 95% interval -8.3 to +0.5.

## Who wrote the wrong query vs who judged it

Share of each writer's wrong queries that each judge passed (lower is better). The diagonal is a model judging its own SQL.

| Writer (wrong queries) | codegemma:7b | codellama:13b | gemma3:12b | gpt-oss:20b | phi4:14b |
|---|---|---|---|---|---|
| codegemma:7b (18) | **100%** | 83% | 39% | 17% | 33% |
| codellama:13b (30) | 97% | **87%** | 23% | 7% | 23% |
| gemma3:12b (17) | 100% | 76% | **41%** | 0% | 47% |
| gpt-oss:20b (10) | 100% | 80% | 80% | **50%** | 80% |
| phi4:14b (14) | 100% | 79% | 50% | 21% | **64%** |

## Choosing the jury automatically by cost

Cost per query, in units of one human review. A wrong query that slips through costs R reviews. Assumes 12% of incoming queries are wrong.

| R | Automatic choice | Paper recipe | Automatic minus recipe (95% interval) | Unanimous jury chosen by cost | Automatic minus that (95% interval) | Best single judge | Human reviews all | Accept all | Usual choice |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.124 | 0.235 | -0.111 [-0.160, -0.068] | 0.124 | +0.000 [+0.000, +0.000] | 0.177 | 1.000 | 0.120 | unanimous: codegemma:7b (5/5 folds) |
| 2 | 0.199 | 0.250 | -0.050 [-0.095, -0.012] | 0.194 | +0.005 [-0.015, +0.025] | 0.194 | 1.000 | 0.240 | confidence: codegemma:7b + gpt-oss:20b (4/5 folds) |
| 5 | 0.246 | 0.294 | -0.049 [-0.081, -0.019] | 0.247 | -0.001 [-0.030, +0.034] | 0.247 | 1.000 | 0.600 | confidence: codegemma:7b + gpt-oss:20b (4/5 folds) |
| 10 | 0.382 | 0.368 | +0.014 [-0.035, +0.079] | 0.374 | +0.009 [-0.042, +0.076] | 0.335 | 1.000 | 1.200 | confidence: gemma3:12b + gpt-oss:20b + phi4:14b (4/5 folds) |
| 20 | 0.341 | 0.517 | -0.175 [-0.378, -0.020] | 0.444 | -0.102 [-0.216, -0.003] | 0.510 | 1.000 | 2.400 | confidence: gemma3:12b + gpt-oss:20b + phi4:14b (5/5 folds) |
| 50 | 0.420 | 0.962 | -0.541 [-1.042, -0.150] | 0.727 | -0.306 [-0.603, -0.050] | 1.036 | 1.000 | 6.000 | confidence: gpt-oss:20b + phi4:14b (4/5 folds) |
| 100 | 0.488 | 1.703 | -1.215 [-2.212, -0.421] | 1.395 | -0.907 [-1.418, -0.485] | 1.912 | 1.000 | 12.000 | confidence: gpt-oss:20b + phi4:14b (4/5 folds) |
