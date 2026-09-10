# Search experiments

Experiments stay on separate branches until they outperform `main` on the
same target with the exact open-source oracle. A failed experiment is kept so
the result can be reproduced without putting unproven logic in a submission.

## Adaptive row/column arms (2026-09-10)

Branch: `experiment/adaptive-arms`

This experiment treated each row or column around a high-scoring complete
molecule as a Thompson-sampling arm. An arm varied one reaction component and
held the others fixed. Twenty percent of arm choices remained uniform, and
18% of the proposal pool remained globally random.

The test used the same 258-residue target and exact Boltz-2 oracle as the
production benchmark: 4 x RTX 4090, 12 warm shards, batches of 48. This has the
same four inference rounds per request as batches of 96 on the competition's
24-shard, 8-GPU oracle.

| Predictions | Submitted `main` | Adaptive arms | Difference |
| ---: | ---: | ---: | ---: |
| 144 | 0.029985 | 0.024664 | -0.005321 |
| 192 | 0.039312 | 0.034517 | -0.004795 |
| 240 | 0.045420 | 0.040032 | -0.005388 |

The arm run was stopped after five requests because the deficit was stable.
Request times were 117-123 seconds; fitting the proposal model took about 0.09
seconds. Search remains oracle-bound, so the lost score was a policy problem,
not a CPU bottleneck.

Lessons:

- Explicitly concentrating requests on currently successful rows and columns
  was worse than the submitted policy's broad row/column completion.
- Exact arm identities turn over as the elite seed set changes. Early arm
  posteriors therefore contain too little evidence to justify concentration.
- A strong individual batch did not repair the portfolio deficit: batch five
  averaged 0.035226 and found a 0.082533 molecule, but its 100-molecule
  portfolio still trailed by 0.005388.
- Keep the submitted early surrogate, broad completion search, and acquisition
  mixture. Future challenger experiments must beat `main` before promotion.

The production branch remained unchanged at commit
`74185be26bb5151ec32dd51db3d546689c7809d4` throughout this test.
