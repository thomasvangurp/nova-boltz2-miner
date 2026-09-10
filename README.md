# NOVA Boltz-2 miner

Source-only redesign of the winning row/column harvest strategy for NOVA's
Boltz-2 Blueprint sandbox.

The submission contains no compiled extension, generated code, model weights,
precomputed score, molecule list, or historical seed. It learns entirely from
the current challenge through the provided oracle socket.

## Search

1. Build and atomically publish a validator-safe random portfolio immediately.
2. Score 192 molecule/target pairs per request: eight balanced rounds across
   the oracle's 24 shards and faster feedback than a cap-sized batch.
3. Complete one reaction axis around component-disjoint current-run winners.
4. Train an ExtraTrees proposal model on current-run observations only.
5. Repeat-score the leading pool and rank by a conservative estimate.
6. Assemble exactly 100 molecules under InChI, Tanimoto, and MACCS constraints.
7. Atomically promote `result.json` after every completed improvement.

CPU proposal construction runs one batch ahead while the blocking GPU oracle
is busy. After the first response, the deadline guard uses measured request
time plus 15 seconds. If the validator still kills the final attempt, the last
atomic checkpoint remains complete.

The entire production runtime is intentionally kept in three readable files:
`miner.py`, `search.py`, and `portfolio.py`.

The validator supplies `nova_miner`, RDKit, NumPy, SciPy, and scikit-learn.
Development dependencies in `pixi.toml` mirror those versions; Pixi is not
invoked inside the competition sandbox.

Run the tests against a checkout of the Blueprint branch:

```sh
PYTHONPATH=/path/to/nova-blueprint/libs pixi run test
```
