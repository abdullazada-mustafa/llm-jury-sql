# Results summary

273 candidate queries (91 correct, 182 wrong) from 40 tasks, reviewed by 5 judges. Everything that is tuned is tuned on other tasks (5-fold, grouped by task); intervals are 95% bootstrap over tasks. Labels: strict.

## Each judge alone

| Judge | Passes correct | Passes wrong | Youden's J | Pattern |
|---|---|---|---|---|
| codegemma:7b | 100% | 90% | 0.10 | rubber stamp |
| codellama:13b | 92% | 58% | 0.35 |  |
| gemma3:12b | 88% | 24% | 0.64 |  |
| gpt-oss:20b | 97% | 3% | 0.93 |  |
| phi4:14b | 93% | 21% | 0.73 |  |

Bug types that most often fool the best single judge (**gpt-oss:20b**), share of those wrong queries it passed:

- distinct_counting (n=7): 29%
- inverted_logic (n=6): 17%
- join_type (n=10): 10%
- null_handling (n=12): 8%
- wrong_boundary (n=29): 3%

## Main comparison

| System | Passes correct | Passes wrong | Youden's J | J vs paper recipe |
|---|---|---|---|---|
| Paper recipe: best unanimous jury, picked on training tasks | 97% | 3% | 0.93 [0.88, 0.98] |  |
| Unanimous, all judges | 78% | 1% | 0.77 [0.68, 0.85] | -16.5 [-23.1, -10.2] |
| Reliability-weighted vote, all judges | 93% | 2% | 0.92 [0.85, 0.97] | -1.6 [-4.5, +0.0] |
| Confidence-weighted vote, all judges | 92% | 2% | 0.90 [0.82, 0.96] | -3.3 [-7.6, +0.0] |
| Confidence-weighted, tuned to the recipe's false accept rate | 93% | 2% | 0.91 [0.83, 0.97] | -2.2 [-5.7, +0.0] |
| Dawid-Skene, all judges, no labels at all | 96% | 13% | 0.83 [0.76, 0.89] | -10.4 [-17.4, -4.0] |

Committee the paper recipe picked: codegemma:7b + gpt-oss:20b (5/5 folds).
Area under the ROC curve for the full panel: weighted 0.981, confidence 0.971, dawid_skene 0.979.

## The veto effect

| Judge | Correct queries it blocks when added | Wrong queries it blocks when added |
|---|---|---|
| codegemma:7b | 0% | 1% |
| codellama:13b | 7% | 26% |
| gemma3:12b | 9% | 46% |
| gpt-oss:20b | 2% | 92% |
| phi4:14b | 5% | 36% |

No judge blocks more correct queries than wrong ones when added. The least useful is **codegemma:7b**, which blocks only 1% of the wrong queries a jury used to pass.

- Per committee, **weighted** vs unanimous (best-J threshold): Youden's J higher in 17/26 committees, clearly higher in 14, clearly lower in 1.
- Per committee, **confidence** vs unanimous (best-J threshold): Youden's J higher in 18/26 committees, clearly higher in 12, clearly lower in 2.

## Choosing the jury automatically by cost

Cost per query, in units of one human review. A wrong query that slips through costs R reviews. Assumes 12% of incoming queries are wrong.

| R | Automatic choice | Paper recipe | Automatic minus recipe (95% interval) | Unanimous jury chosen by cost | Automatic minus that (95% interval) | Best single judge | Human reviews all | Accept all | Usual choice |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.124 | 0.159 | -0.035 [-0.073, -0.006] | 0.124 | +0.000 [+0.000, +0.000] | 0.155 | 1.000 | 0.120 | unanimous: codegemma:7b (5/5 folds) |
| 2 | 0.175 | 0.163 | +0.012 [+0.005, +0.022] | 0.159 | +0.016 [+0.009, +0.026] | 0.159 | 1.000 | 0.240 | confidence: codellama:13b + gpt-oss:20b + phi4:14b (3/5 folds) |
| 5 | 0.171 | 0.175 | -0.004 [-0.004, -0.004] | 0.171 | +0.000 [+0.000, +0.000] | 0.171 | 1.000 | 0.600 | unanimous: gpt-oss:20b (5/5 folds) |
| 10 | 0.202 | 0.194 | +0.008 [-0.002, +0.028] | 0.191 | +0.012 [+0.002, +0.032] | 0.191 | 1.000 | 1.200 | unanimous: gpt-oss:20b (3/5 folds) |
| 20 | 0.254 | 0.234 | +0.020 [-0.009, +0.049] | 0.230 | +0.023 [-0.005, +0.053] | 0.230 | 1.000 | 2.400 | weighted: codellama:13b + gemma3:12b + gpt-oss:20b (3/5 folds) |
| 50 | 0.313 | 0.353 | -0.039 [-0.156, +0.056] | 0.389 | -0.075 [-0.177, -0.002] | 0.349 | 1.000 | 6.000 | weighted: codellama:13b + gemma3:12b + gpt-oss:20b + phi4:14b (2/5 folds) |
| 100 | 0.457 | 0.550 | -0.093 [-0.361, +0.087] | 0.443 | +0.014 [+0.004, +0.034] | 0.547 | 1.000 | 12.000 | confidence: gemma3:12b + gpt-oss:20b (2/5 folds) |
