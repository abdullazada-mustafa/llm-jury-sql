# Results summary

397 candidate queries (304 correct, 93 wrong) from 82 tasks, reviewed by 5 judges. Everything that is tuned is tuned on other tasks (5-fold, grouped by task); intervals are 95% bootstrap over tasks. Labels: strict.

## Each judge alone

| Judge | Passes correct | Passes wrong | Youden's J | Pattern |
|---|---|---|---|---|
| codegemma:7b | 100% | 99% | 0.01 | rubber stamp |
| codellama:13b | 96% | 83% | 0.13 |  |
| gemma3:12b | 92% | 42% | 0.50 |  |
| gpt-oss:20b | 95% | 16% | 0.79 |  |
| phi4:14b | 96% | 45% | 0.51 |  |

Bug types that most often fool the best single judge (**gpt-oss:20b**), share of those wrong queries it passed:

- natural (n=93): 16%

## Main comparison

| System | Passes correct | Passes wrong | Youden's J | J vs paper recipe |
|---|---|---|---|---|
| Paper recipe: best unanimous jury, picked on training tasks | 91% | 15% | 0.76 [0.67, 0.85] |  |
| Unanimous, all judges | 85% | 9% | 0.77 [0.68, 0.84] | +0.2 [-6.3, +8.6] |
| Reliability-weighted vote, all judges | 89% | 13% | 0.76 [0.67, 0.84] | -0.5 [-4.3, +3.6] |
| Confidence-weighted vote, all judges | 88% | 14% | 0.74 [0.65, 0.82] | -2.5 [-7.4, +2.2] |
| Confidence-weighted, tuned to the recipe's false accept rate | 90% | 16% | 0.74 [0.63, 0.83] | -2.4 [-9.5, +3.3] |
| Dawid-Skene, all judges, no labels at all | 93% | 31% | 0.62 [0.48, 0.74] | -14.5 [-29.3, -2.1] |

Committee the paper recipe picked: codegemma:7b + gpt-oss:20b (3/5 folds), gpt-oss:20b + phi4:14b (1/5 folds), gemma3:12b + gpt-oss:20b + phi4:14b (1/5 folds).
Area under the ROC curve for the full panel: weighted 0.904, confidence 0.954, dawid_skene 0.898.

## The veto effect

| Judge | Correct queries it blocks when added | Wrong queries it blocks when added |
|---|---|---|
| codegemma:7b | 0% | 0% |
| codellama:13b | 3% | 9% |
| gemma3:12b | 6% | 38% |
| gpt-oss:20b | 4% | 73% |
| phi4:14b | 2% | 26% |

No judge blocks more correct queries than wrong ones when added. The least useful is **codegemma:7b**, which blocks only 0% of the wrong queries a jury used to pass.

- Per committee, **weighted** vs unanimous (best-J threshold): Youden's J higher in 8/26 committees, clearly higher in 0, clearly lower in 0.
- Per committee, **confidence** vs unanimous (best-J threshold): Youden's J higher in 21/26 committees, clearly higher in 3, clearly lower in 0.

## Weights learned per SQL-writing model ("adaptive")

| Writer model | Wrong queries | J, pooled weights | J, per-writer weights |
|---|---|---|---|
| codegemma:7b | 21 | 0.74 | 0.66 |
| codellama:13b | 30 | 0.81 | 0.77 |
| gemma3:12b | 17 | 0.86 | 0.72 |
| gpt-oss:20b | 10 | 0.49 | 0.52 |
| phi4:14b | 15 | 0.72 | 0.74 |
| (all) | 93 | 0.76 | 0.72 |

Overall difference (per-writer minus pooled): -4.4 points, 95% interval -9.4 to -0.1.

## Who wrote the wrong query vs who judged it

Share of each writer's wrong queries that each judge passed (lower is better). The diagonal is a model judging its own SQL.

| Writer (wrong queries) | codegemma:7b | codellama:13b | gemma3:12b | gpt-oss:20b | phi4:14b |
|---|---|---|---|---|---|
| codegemma:7b (21) | **100%** | 86% | 43% | 19% | 43% |
| codellama:13b (30) | 97% | **87%** | 23% | 7% | 23% |
| gemma3:12b (17) | 100% | 76% | **41%** | 0% | 47% |
| gpt-oss:20b (10) | 100% | 80% | 80% | **50%** | 80% |
| phi4:14b (15) | 100% | 80% | 53% | 27% | **67%** |

## Choosing the jury automatically by cost

Cost per query, in units of one human review. A wrong query that slips through costs R reviews. Assumes 12% of incoming queries are wrong.

| R | Automatic choice | Paper recipe | Automatic minus recipe (95% interval) | Unanimous jury chosen by cost | Automatic minus that (95% interval) | Best single judge | Human reviews all | Accept all | Usual choice |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.124 | 0.209 | -0.085 [-0.130, -0.048] | 0.124 | +0.000 [+0.000, +0.000] | 0.172 | 1.000 | 0.120 | unanimous: codegemma:7b (5/5 folds) |
| 2 | 0.189 | 0.227 | -0.038 [-0.082, -0.001] | 0.191 | -0.002 [-0.023, +0.018] | 0.191 | 1.000 | 0.240 | confidence: codegemma:7b + gpt-oss:20b (3/5 folds) |
| 5 | 0.275 | 0.281 | -0.006 [-0.036, +0.028] | 0.249 | +0.026 [-0.009, +0.071] | 0.249 | 1.000 | 0.600 | confidence: codegemma:7b + gpt-oss:20b (2/5 folds) |
| 10 | 0.382 | 0.371 | +0.011 [-0.023, +0.044] | 0.387 | -0.005 [-0.031, +0.015] | 0.346 | 1.000 | 1.200 | confidence: gemma3:12b + gpt-oss:20b + phi4:14b (2/5 folds) |
| 20 | 0.412 | 0.552 | -0.140 [-0.322, +0.011] | 0.548 | -0.135 [-0.307, +0.000] | 0.540 | 1.000 | 2.400 | confidence: gemma3:12b + gpt-oss:20b + phi4:14b (4/5 folds) |
| 50 | 0.598 | 1.094 | -0.496 [-0.958, -0.113] | 0.829 | -0.231 [-0.480, -0.008] | 1.120 | 1.000 | 6.000 | confidence: gemma3:12b + gpt-oss:20b + phi4:14b (3/5 folds) |
| 100 | 0.716 | 1.997 | -1.282 [-2.198, -0.494] | 1.524 | -0.808 [-1.247, -0.429] | 2.088 | 1.000 | 12.000 | confidence: gemma3:12b + gpt-oss:20b + phi4:14b (3/5 folds) |
