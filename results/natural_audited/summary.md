# Results summary

397 candidate queries (314 correct, 83 wrong) from 82 tasks, reviewed by 5 judges. Everything that is tuned is tuned on other tasks (5-fold, grouped by task); intervals are 95% bootstrap over tasks. Labels: audited.

## Each judge alone

| Judge | Passes correct | Passes wrong | Youden's J | Pattern |
|---|---|---|---|---|
| codegemma:7b | 100% | 99% | 0.01 | rubber stamp |
| codellama:13b | 96% | 81% | 0.15 |  |
| gemma3:12b | 91% | 39% | 0.53 |  |
| gpt-oss:20b | 94% | 11% | 0.83 |  |
| phi4:14b | 96% | 39% | 0.58 |  |

Bug types that most often fool the best single judge (**gpt-oss:20b**), share of those wrong queries it passed:

- natural (n=83): 11%

## Main comparison

| System | Passes correct | Passes wrong | Youden's J | J vs paper recipe |
|---|---|---|---|---|
| Paper recipe: best unanimous jury, picked on training tasks | 92% | 11% | 0.82 [0.74, 0.88] |  |
| Unanimous, all judges | 84% | 5% | 0.79 [0.71, 0.86] | -2.6 [-10.1, +6.6] |
| Reliability-weighted vote, all judges | 90% | 10% | 0.81 [0.72, 0.88] | -0.7 [-5.1, +3.3] |
| Confidence-weighted vote, all judges | 89% | 18% | 0.70 [0.56, 0.82] | -11.1 [-25.8, +0.6] |
| Confidence-weighted, tuned to the recipe's false accept rate | 90% | 11% | 0.79 [0.69, 0.88] | -2.2 [-10.3, +4.2] |
| Dawid-Skene, all judges, no labels at all | 93% | 24% | 0.69 [0.54, 0.82] | -12.6 [-29.1, +1.4] |

Committee the paper recipe picked: codegemma:7b + gpt-oss:20b (3/5 folds), gpt-oss:20b + phi4:14b (2/5 folds).
Area under the ROC curve for the full panel: weighted 0.946, confidence 0.968, dawid_skene 0.933.

## The veto effect

| Judge | Correct queries it blocks when added | Wrong queries it blocks when added |
|---|---|---|
| codegemma:7b | 0% | 0% |
| codellama:13b | 3% | 13% |
| gemma3:12b | 7% | 40% |
| gpt-oss:20b | 5% | 82% |
| phi4:14b | 2% | 34% |

No judge blocks more correct queries than wrong ones when added. The least useful is **codegemma:7b**, which blocks only 0% of the wrong queries a jury used to pass.

- Per committee, **weighted** vs unanimous (best-J threshold): Youden's J higher in 10/26 committees, clearly higher in 0, clearly lower in 0.
- Per committee, **confidence** vs unanimous (best-J threshold): Youden's J higher in 12/26 committees, clearly higher in 1, clearly lower in 4.

## Weights learned per SQL-writing model ("adaptive")

| Writer model | Wrong queries | J, pooled weights | J, per-writer weights |
|---|---|---|---|
| codegemma:7b | 16 | 0.74 | 0.64 |
| codellama:13b | 30 | 0.83 | 0.77 |
| gemma3:12b | 17 | 0.86 | 0.72 |
| gpt-oss:20b | 8 | 0.56 | 0.53 |
| phi4:14b | 12 | 0.87 | 0.89 |
| (all) | 83 | 0.81 | 0.74 |

Overall difference (per-writer minus pooled): -6.6 points, 95% interval -15.7 to +1.2.

## Who wrote the wrong query vs who judged it

Share of each writer's wrong queries that each judge passed (lower is better). The diagonal is a model judging its own SQL.

| Writer (wrong queries) | codegemma:7b | codellama:13b | gemma3:12b | gpt-oss:20b | phi4:14b |
|---|---|---|---|---|---|
| codegemma:7b (16) | **100%** | 81% | 38% | 12% | 25% |
| codellama:13b (30) | 97% | **87%** | 23% | 7% | 23% |
| gemma3:12b (17) | 100% | 76% | **41%** | 0% | 47% |
| gpt-oss:20b (8) | 100% | 75% | 75% | **50%** | 75% |
| phi4:14b (12) | 100% | 75% | 50% | 8% | **58%** |

## Choosing the jury automatically by cost

Cost per query, in units of one human review. A wrong query that slips through costs R reviews. Assumes 12% of incoming queries are wrong.

| R | Automatic choice | Paper recipe | Automatic minus recipe (95% interval) | Unanimous jury chosen by cost | Automatic minus that (95% interval) | Best single judge | Human reviews all | Accept all | Usual choice |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.124 | 0.200 | -0.076 [-0.110, -0.045] | 0.124 | +0.000 [+0.000, +0.000] | 0.182 | 1.000 | 0.120 | unanimous: codegemma:7b (5/5 folds) |
| 2 | 0.177 | 0.213 | -0.036 [-0.068, -0.006] | 0.195 | -0.018 [-0.041, +0.005] | 0.195 | 1.000 | 0.240 | confidence: codegemma:7b + gpt-oss:20b (4/5 folds) |
| 5 | 0.223 | 0.252 | -0.029 [-0.048, -0.009] | 0.234 | -0.010 [-0.024, +0.001] | 0.234 | 1.000 | 0.600 | confidence: codegemma:7b + gpt-oss:20b (5/5 folds) |
| 10 | 0.339 | 0.317 | +0.023 [+0.003, +0.055] | 0.316 | +0.024 [+0.004, +0.057] | 0.299 | 1.000 | 1.200 | confidence: codellama:13b + gpt-oss:20b (2/5 folds) |
| 20 | 0.422 | 0.447 | -0.025 [-0.134, +0.069] | 0.456 | -0.034 [-0.109, +0.020] | 0.429 | 1.000 | 2.400 | confidence: gemma3:12b + gpt-oss:20b + phi4:14b (4/5 folds) |
| 50 | 0.438 | 0.837 | -0.399 [-0.838, -0.036] | 0.764 | -0.326 [-0.740, -0.000] | 0.819 | 1.000 | 6.000 | confidence: gpt-oss:20b + phi4:14b (4/5 folds) |
| 100 | 0.510 | 1.488 | -0.978 [-1.843, -0.232] | 1.270 | -0.759 [-1.590, -0.101] | 1.470 | 1.000 | 12.000 | confidence: gpt-oss:20b + phi4:14b (4/5 folds) |
