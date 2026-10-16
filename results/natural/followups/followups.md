# Follow-up analyses (strict labels)

397 model-written queries (93 wrong). Each section answers a sentence from Ullah and Serwadda (2026). Intervals: 95% bootstrap over questions.

## 1. Shared mistakes

> "errors across judges can be correlated (and simple majorities can amplify a shared bias)" (Section I)

| Jury | Rule | Wrong queries passed | If mistakes were independent | Ratio [95% interval] |
|---|---|---|---|---|
| all five | unanimous | 9% | 2.5% | 3.4x [1.6, 6.7] |
| all five | majority | 57% | 64.9% | 0.9x [0.8, 1.0] |
| gemma3 + gpt-oss + phi4 | unanimous | 9% | 3.1% | 2.8x [1.3, 5.4] |
| gemma3 + gpt-oss + phi4 | majority | 37% | 26.9% | 1.4x [1.1, 1.7] |
| gpt-oss + phi4 | unanimous | 14% | 7.3% | 1.9x [1.4, 2.6] |
| gemma3 + gpt-oss | unanimous | 11% | 6.8% | 1.6x [0.9, 2.4] |
| gemma3 + phi4 | unanimous | 29% | 18.9% | 1.5x [1.3, 1.9] |

When gpt-oss is fooled, how often the others are fooled too (vs their usual rate):

- codellama:13b: 87% (usually 83%)
- gemma3:12b: 67% (usually 42%)
- phi4:14b: 87% (usually 45%)

| System | Passes correct | Passes wrong | J | J vs recipe [95%] | Wrong passed vs recipe [95%] |
|---|---|---|---|---|---|
| Paper recipe (best unanimous jury) | 91% | 15% | 0.76 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| Unanimous, all five | 85% | 9% | 0.77 | +0.2 [-6.3, +8.6] | -6.5 [-13.2, -1.4] |
| Majority, all five (3 of 5) | 100% | 57% | 0.43 | -33.7 [-46.4, -20.8] | +41.9 [+29.6, +54.0] |
| Unanimous, gemma3 + gpt-oss + phi4 | 86% | 9% | 0.78 | +1.2 [-5.1, +9.2] | -6.5 [-13.2, -1.4] |
| Majority, gemma3 + gpt-oss + phi4 (2 of 3) | 98% | 37% | 0.61 | -15.3 [-29.2, -3.2] | +21.5 [+10.5, +34.7] |

## 2. Is a model easier on its own SQL?

> "self-judging may introduce optimism or style bias" (Section III-C)

Extra chance, in percentage points, that a judge passes a query, after removing how hard the query is and how lenient the judge is in general:

| | Wrong queries [95%] | Correct queries [95%] |
|---|---|---|
| Judge wrote this query | +22.1 [+11.5, +32.0] | -0.1 [-1.7, +1.4] |
| Judge did not write it, but got the same question wrong | +18.0 [+5.7, +30.4] | -2.1 [-8.8, +3.8] |
| Difference: own text vs same misunderstanding | +4.0 [-5.3, +13.0] | +2.0 [-4.1, +9.2] |

| Judge | Its own wrong queries | It passed | Other judges passed the same queries |
|---|---|---|---|
| codegemma:7b | 21 | 100% | 48% |
| codellama:13b | 30 | 87% | 38% |
| gemma3:12b | 17 | 41% | 56% |
| gpt-oss:20b | 10 | 50% | 85% |
| phi4:14b | 15 | 67% | 65% |

## 3. One jury per writer model, and keeping the writer off its own jury

> "This argues for per-generator validation" ... "deployments should test both inclusive (with self-judging) and exclusive configurations" (Section V)

Juries are chosen on training questions and scored on held-out questions (the paper chose them on the same queries it reported).

| Comparison | J difference [95%] | Wrong passed difference [95%] |
|---|---|---|
| per-writer choice vs one shared choice | +4.1 [-1.4, +9.8] | -2.2 [-6.5, +2.0] |
| writer kept off its own jury vs allowed | -1.4 [-6.5, +3.6] | +1.1 [-3.6, +5.8] |
| same, with per-writer choice | -5.9 [-10.2, -1.9] | +4.3 [+1.0, +8.4] |

| Writer | Wrong | J: shared choice | J: per-writer choice | J: shared, writer excluded | J: per-writer, writer excluded |
|---|---|---|---|---|---|
| codegemma:7b | 21 | 0.77 | 0.73 | 0.72 | 0.69 |
| codellama:13b | 30 | 0.83 | 0.85 | 0.83 | 0.85 |
| gemma3:12b | 17 | 0.86 | 0.92 | 0.88 | 0.92 |
| gpt-oss:20b | 10 | 0.43 | 0.66 | 0.32 | 0.22 |
| phi4:14b | 15 | 0.69 | 0.75 | 0.72 | 0.75 |
| (all writers) | 93 | 0.76 | 0.81 | 0.75 | 0.75 |

## 4. Harder questions

> future work: "adding harder compositional queries" (Section V)

| Questions | Answers | Did not run | Correct | Wrong | Wrong passed by paper recipe | Wrong passed by gpt-oss |
|---|---|---|---|---|---|---|
| original 40 | 200 | 2 | 187 | 11 | 27% | 27% |
| harder 42 | 210 | 11 | 117 | 82 | 13% | 15% |
- harder minus original, wrong share: +35.7 points [+24.8, +46.4]
- harder minus original, fpr paper recipe: -13.9 points [-61.4, +16.7]
- harder minus original, fpr confidence all: -15.1 points [-62.9, +15.9]
- harder minus original, fpr gpt-oss: -12.6 points [-59.9, +17.9]

## 5. Speed versus safety

> "we do not optimize for latency or cost" (Section V)

- Stopping a unanimous jury at the first INCORRECT gives the same decisions and saves 9% of judge time on average over all juries of 2+ judges (range 2% to 16%).
- All five judges: 25.4 s per query asking everyone, 21.4 s with stopping.
- Best unanimous jury (in-sample): gpt-oss:20b, J 0.79, 6.9 s per query.
- Fastest jury within 0.02 of that J: gpt-oss:20b, J 0.79, 6.9 s per query.

Fastest unanimous jury that keeps wrong queries passed under a target (in-sample, as in the paper):

| Target: wrong passed at most | Fastest jury | Passes correct | Passes wrong | Seconds per query |
|---|---|---|---|---|
| 20% | gpt-oss:20b | 95% | 16% | 6.9 |
| 15% | gpt-oss:20b + phi4:14b | 92% | 14% | 11.1 |
| 10% | gemma3:12b + gpt-oss:20b + phi4:14b | 86% | 9% | 14.9 |

Fastest juries for each level of quality (Pareto front):

| Jury | J | Seconds per query (stop at first NO) | Asking every judge |
|---|---|---|---|
| codegemma:7b | 0.01 | 3.5 | 3.5 |
| phi4:14b | 0.51 | 4.7 | 4.7 |
| gpt-oss:20b | 0.79 | 6.9 | 6.9 |

## 6. A jury that learns as checked queries arrive

> future work: "adaptive or weighted committees learned from per-judge calibration" (Section V)

Questions arrive in random order (40 orders). After k questions are checked, each method is fitted on them and scored on the questions not yet seen. Mean over orders.

| Questions checked | Queries checked | J: unanimous all | J: paper recipe | J: weighted | Cost R=50: paper recipe | Cost R=50: chosen for cost |
|---|---|---|---|---|---|---|
| 5 | 24 | 0.76 | 0.72 | 0.71 | 1.41 | 1.66 |
| 10 | 48 | 0.76 | 0.73 | 0.72 | 1.35 | 1.45 |
| 15 | 73 | 0.76 | 0.75 | 0.74 | 1.13 | 1.23 |
| 20 | 97 | 0.76 | 0.76 | 0.74 | 1.05 | 1.01 |
| 30 | 145 | 0.76 | 0.75 | 0.74 | 1.12 | 0.75 |
| 40 | 193 | 0.76 | 0.74 | 0.74 | 1.15 | 0.66 |
| 50 | 242 | 0.76 | 0.75 | 0.75 | 1.14 | 0.63 |
| 60 | 290 | 0.77 | 0.73 | 0.74 | 1.16 | 0.60 |
