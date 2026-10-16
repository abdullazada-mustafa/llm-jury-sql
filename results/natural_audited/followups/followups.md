# Follow-up analyses (audited labels)

397 model-written queries (83 wrong). Each section answers a sentence from Ullah and Serwadda (2026). Intervals: 95% bootstrap over questions.

## 1. Shared mistakes

> "errors across judges can be correlated (and simple majorities can amplify a shared bias)" (Section I)

| Jury | Rule | Wrong queries passed | If mistakes were independent | Ratio [95% interval] |
|---|---|---|---|---|
| all five | unanimous | 5% | 1.3% | 3.7x [0.8, 9.5] |
| all five | majority | 52% | 56.9% | 0.9x [0.8, 1.0] |
| gemma3 + gpt-oss + phi4 | unanimous | 5% | 1.6% | 3.0x [0.6, 7.6] |
| gemma3 + gpt-oss + phi4 | majority | 30% | 20.0% | 1.5x [1.2, 2.0] |
| gpt-oss + phi4 | unanimous | 8% | 4.2% | 2.0x [1.2, 3.1] |
| gemma3 + gpt-oss | unanimous | 7% | 4.2% | 1.7x [0.8, 3.2] |
| gemma3 + phi4 | unanimous | 24% | 14.9% | 1.6x [1.3, 2.1] |

When gpt-oss is fooled, how often the others are fooled too (vs their usual rate):

- codellama:13b: 78% (usually 81%)
- gemma3:12b: 67% (usually 39%)
- phi4:14b: 78% (usually 39%)

| System | Passes correct | Passes wrong | J | J vs recipe [95%] | Wrong passed vs recipe [95%] |
|---|---|---|---|---|---|
| Paper recipe (best unanimous jury) | 92% | 11% | 0.82 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| Unanimous, all five | 84% | 5% | 0.79 | -2.6 [-10.1, +6.6] | -6.0 [-13.2, -1.2] |
| Majority, all five (3 of 5) | 100% | 52% | 0.48 | -33.6 [-47.3, -19.4] | +41.0 [+27.6, +54.1] |
| Unanimous, gemma3 + gpt-oss + phi4 | 85% | 5% | 0.80 | -1.6 [-8.9, +7.4] | -6.0 [-13.2, -1.2] |
| Majority, gemma3 + gpt-oss + phi4 (2 of 3) | 97% | 30% | 0.67 | -14.2 [-28.9, -1.9] | +19.3 [+7.3, +33.7] |

## 2. Is a model easier on its own SQL?

> "self-judging may introduce optimism or style bias" (Section III-C)

Extra chance, in percentage points, that a judge passes a query, after removing how hard the query is and how lenient the judge is in general:

| | Wrong queries [95%] | Correct queries [95%] |
|---|---|---|
| Judge wrote this query | +21.6 [+10.2, +32.3] | +0.2 [-1.6, +1.8] |
| Judge did not write it, but got the same question wrong | +14.8 [+2.4, +27.6] | -1.4 [-8.5, +5.1] |
| Difference: own text vs same misunderstanding | +6.8 [-3.8, +17.0] | +1.6 [-5.0, +8.9] |

| Judge | Its own wrong queries | It passed | Other judges passed the same queries |
|---|---|---|---|
| codegemma:7b | 16 | 100% | 39% |
| codellama:13b | 30 | 87% | 38% |
| gemma3:12b | 17 | 41% | 56% |
| gpt-oss:20b | 8 | 50% | 81% |
| phi4:14b | 12 | 58% | 58% |

## 3. One jury per writer model, and keeping the writer off its own jury

> "This argues for per-generator validation" ... "deployments should test both inclusive (with self-judging) and exclusive configurations" (Section V)

Juries are chosen on training questions and scored on held-out questions (the paper chose them on the same queries it reported).

| Comparison | J difference [95%] | Wrong passed difference [95%] |
|---|---|---|
| per-writer choice vs one shared choice | +3.4 [+0.3, +7.6] | -2.4 [-6.3, +0.0] |
| writer kept off its own jury vs allowed | -0.6 [-5.8, +4.9] | +0.0 [-5.0, +4.7] |
| same, with per-writer choice | -4.6 [-8.9, -0.7] | +3.6 [+0.0, +7.7] |

| Writer | Wrong | J: shared choice | J: per-writer choice | J: shared, writer excluded | J: per-writer, writer excluded |
|---|---|---|---|---|---|
| codegemma:7b | 16 | 0.77 | 0.78 | 0.75 | 0.77 |
| codellama:13b | 30 | 0.85 | 0.85 | 0.85 | 0.85 |
| gemma3:12b | 17 | 0.88 | 0.92 | 0.88 | 0.92 |
| gpt-oss:20b | 8 | 0.45 | 0.70 | 0.42 | 0.29 |
| phi4:14b | 12 | 0.89 | 0.89 | 0.90 | 0.89 |
| (all writers) | 83 | 0.82 | 0.85 | 0.81 | 0.80 |

## 4. Harder questions

> future work: "adding harder compositional queries" (Section V)

| Questions | Answers | Did not run | Correct | Wrong | Wrong passed by paper recipe | Wrong passed by gpt-oss |
|---|---|---|---|---|---|---|
| original 40 | 200 | 2 | 190 | 8 | 0% | 0% |
| harder 42 | 210 | 11 | 124 | 75 | 12% | 12% |
- harder minus original, wrong share: +33.6 points [+23.3, +43.7]
- harder minus original, fpr paper recipe: +12.0 points [+4.9, +20.3]
- harder minus original, fpr confidence all: +6.2 points [-42.2, +28.8]
- harder minus original, fpr gpt-oss: +12.0 points [+4.9, +20.3]

## 5. Speed versus safety

> "we do not optimize for latency or cost" (Section V)

- Stopping a unanimous jury at the first INCORRECT gives the same decisions and saves 10% of judge time on average over all juries of 2+ judges (range 2% to 17%).
- All five judges: 25.4 s per query asking everyone, 21.2 s with stopping.
- Best unanimous jury (in-sample): gpt-oss:20b, J 0.83, 6.9 s per query.
- Fastest jury within 0.02 of that J: gpt-oss:20b, J 0.83, 6.9 s per query.

Fastest unanimous jury that keeps wrong queries passed under a target (in-sample, as in the paper):

| Target: wrong passed at most | Fastest jury | Passes correct | Passes wrong | Seconds per query |
|---|---|---|---|---|
| 20% | gpt-oss:20b | 94% | 11% | 6.9 |
| 15% | gpt-oss:20b | 94% | 11% | 6.9 |
| 10% | gpt-oss:20b + phi4:14b | 91% | 8% | 11.0 |

Fastest juries for each level of quality (Pareto front):

| Jury | J | Seconds per query (stop at first NO) | Asking every judge |
|---|---|---|---|
| codegemma:7b | 0.01 | 3.5 | 3.5 |
| phi4:14b | 0.58 | 4.7 | 4.7 |
| gpt-oss:20b | 0.83 | 6.9 | 6.9 |

## 6. A jury that learns as checked queries arrive

> future work: "adaptive or weighted committees learned from per-judge calibration" (Section V)

Questions arrive in random order (40 orders). After k questions are checked, each method is fitted on them and scored on the questions not yet seen. Mean over orders.

| Questions checked | Queries checked | J: unanimous all | J: paper recipe | J: weighted | Cost R=50: paper recipe | Cost R=50: chosen for cost |
|---|---|---|---|---|---|---|
| 5 | 24 | 0.79 | 0.74 | 0.73 | 1.30 | 1.54 |
| 10 | 48 | 0.79 | 0.74 | 0.75 | 1.23 | 1.41 |
| 15 | 73 | 0.78 | 0.77 | 0.76 | 1.01 | 1.12 |
| 20 | 97 | 0.78 | 0.78 | 0.75 | 0.93 | 1.05 |
| 30 | 145 | 0.78 | 0.78 | 0.77 | 0.92 | 0.81 |
| 40 | 193 | 0.78 | 0.78 | 0.77 | 0.94 | 0.64 |
| 50 | 242 | 0.79 | 0.79 | 0.79 | 0.89 | 0.61 |
| 60 | 290 | 0.79 | 0.79 | 0.78 | 0.90 | 0.58 |
