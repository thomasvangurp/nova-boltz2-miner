from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Event

from rdkit import DataStructs

import search as search_module
from portfolio import maccs_entropy, select_portfolio, write_result
from search import (
    CandidateStore,
    Challenge,
    OracleScorer,
    ReactionSpace,
    Search,
    Settings,
    load_challenge,
)


BLUEPRINT_DB = (
    Path(__import__("nova_miner").__file__).resolve().parent
    / "combinatorial_db"
    / "molecules.sqlite"
)


def challenge(num_molecules: int = 100) -> Challenge:
    return Challenge(
        targets=("ACDEFGHIKLMNPQRSTVWY" * 4,),
        antitargets=(),
        antitarget_weight=0.9,
        num_molecules=num_molecules,
        min_heavy_atoms=20,
        min_rotatable_bonds=1,
        max_rotatable_bonds=10,
        entropy_threshold=0.25,
        tanimoto_threshold=0.9,
        allowed_reaction=None,
    )


class FakeOracle:
    """Fast deterministic stand-in for the Unix-socket oracle."""

    def __init__(self):
        self.calls = 0

    def score(self, targets: list[str], smiles: list[str]) -> list[dict]:
        self.calls += 1
        rows = []
        for smiles_value in smiles:
            scores = []
            for target in targets:
                digest = hashlib.sha256(f"{target}|{smiles_value}".encode()).digest()
                probability = 0.4 + int.from_bytes(digest[:2], "big") / 131_072
                affinity = -0.5 - int.from_bytes(digest[2:4], "big") / 16_384
                scores.append(
                    {
                        "affinity_probability_binary": probability,
                        "affinity_pred_value": affinity,
                    }
                )
            rows.append({"smiles": smiles_value, "scores": scores})
        return rows


def make_candidates(count: int, cfg: Challenge | None = None):
    cfg = cfg or challenge()
    space = ReactionSpace(str(BLUEPRINT_DB), cfg.allowed_reaction)
    store = CandidateStore(cfg)
    names = space.random_names(__import__("random").Random(68), count * 12)
    candidates = store.materialize(names, count)
    assert len(candidates) == count
    return space, store, candidates


def test_loads_current_blueprint_input(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(
        json.dumps(
            {
                "config": {
                    "antitarget_weight": 0.9,
                    "num_molecules": 100,
                    "min_heavy_atoms": 20,
                    "min_rotatable_bonds": 1,
                    "max_rotatable_bonds": 10,
                    "entropy_min_threshold": 0.25,
                    "tanimoto_max_threshold": 0.9,
                },
                "challenge": {
                    "target_sequences": ["ACDEFGHIK"],
                    "antitarget_sequences": [],
                    "allowed_reaction": "rxn:4",
                },
            }
        )
    )
    loaded = load_challenge(str(path))
    assert loaded.num_molecules == 100
    assert loaded.allowed_reaction == 4


def test_reaction_space_and_materialization_match_validator_constraints():
    space, _, candidates = make_candidates(120)
    assert {reaction.reaction_id for reaction in space.reactions} == {1, 2, 3, 4, 5}
    assert len({candidate.name for candidate in candidates}) == 120
    assert len({candidate.inchikey for candidate in candidates}) == 120
    assert all(candidate.heavy_atoms >= 20 for candidate in candidates)

    restricted = ReactionSpace(str(BLUEPRINT_DB), allowed_reaction=4)
    assert {reaction.reaction_id for reaction in restricted.reactions} == {4}
    assert all(
        name.startswith("rxn:4:")
        for name in restricted.random_names(__import__("random").Random(68), 20)
    )


def test_oracle_batching_and_score_formula():
    cfg = challenge()
    _, _, candidates = make_candidates(12, cfg)
    scorer = OracleScorer(
        cfg,
        Settings(prediction_batch=8),
        "/unused",
        oracle=FakeOracle(),
    )
    accepted, _ = scorer.score(candidates)
    assert accepted == 8
    assert scorer.predictions == 8
    assert len(scorer.last_scores) == 8
    assert all(len(candidate.observations) == 1 for candidate in candidates[:8])


def test_production_defaults_favour_adaptive_discovery(monkeypatch):
    for name in (
        "NOVA_DISCOVERY_FRACTION",
        "NOVA_PREDICTION_BATCH",
        "NOVA_PROPOSAL_MULTIPLIER",
        "NOVA_MIN_SURROGATE_SAMPLES",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_environment()
    assert settings.discovery_fraction == 0.84
    assert settings.prediction_batch == 96
    assert settings.proposal_multiplier == 8
    assert settings.min_surrogate_samples == 96


def test_portfolio_is_exact_diverse_and_entropy_safe(tmp_path):
    cfg = challenge()
    _, store, candidates = make_candidates(240, cfg)
    scorer = OracleScorer(
        cfg,
        Settings(prediction_batch=240),
        "/unused",
        oracle=FakeOracle(),
    )
    scorer.score(candidates)
    selected, _, entropy = select_portfolio(
        store.scored(), 100, 0.9, 0.256, store.noise_floor()
    )
    assert len(selected) == 100
    assert len({candidate.inchikey for candidate in selected}) == 100
    assert entropy > 0.256
    for index, candidate in enumerate(selected):
        similarities = DataStructs.BulkTanimotoSimilarity(
            candidate.morgan, [other.morgan for other in selected[index + 1 :]]
        )
        assert max(similarities, default=0.0) < 0.9
    assert (
        abs(entropy - maccs_entropy([candidate.maccs for candidate in selected]))
        < 1e-12
    )

    result_path = tmp_path / "result.json"
    write_result(result_path, selected)
    payload = json.loads(result_path.read_text())
    assert payload == {"molecules": [candidate.name for candidate in selected]}
    assert not list(tmp_path.glob("*.tmp"))


def test_short_end_to_end_run_preserves_valid_checkpoint(tmp_path):
    cfg = challenge()
    settings = Settings(
        runtime_seconds=1_000,
        shutdown_guard_seconds=0,
        prediction_batch=120,
        proposal_multiplier=2,
        min_surrogate_samples=100,
        surrogate_refit_every=1,
        max_iterations=3,
    )
    result_path = tmp_path / "result.json"
    search = Search(
        cfg,
        settings,
        str(result_path),
        "/unused",
        oracle=FakeOracle(),
        db_path=str(BLUEPRINT_DB),
    )
    search.run()
    result = json.loads(result_path.read_text())
    assert len(result["molecules"]) == 100
    assert len(set(result["molecules"])) == 100
    assert search.scorer.predictions == 360
    assert search.surrogate.fitted


def test_surrogate_fits_as_soon_as_the_sample_threshold_is_reached(tmp_path):
    cfg = challenge(num_molecules=20)
    search = Search(
        cfg,
        Settings(
            runtime_seconds=1_000,
            shutdown_guard_seconds=0,
            prediction_batch=24,
            proposal_multiplier=2,
            min_surrogate_samples=40,
            surrogate_refit_every=3,
            max_iterations=2,
        ),
        str(tmp_path / "result.json"),
        "/unused",
        oracle=FakeOracle(),
        db_path=str(BLUEPRINT_DB),
    )
    search.run()
    assert search.scorer.predictions == 48
    assert search.surrogate.fitted


def test_worse_portfolio_never_replaces_the_published_checkpoint(
    tmp_path, monkeypatch
):
    cfg = challenge(num_molecules=2)
    result_path = tmp_path / "result.json"
    search = Search(
        cfg,
        Settings(prediction_batch=4),
        str(result_path),
        "/unused",
        oracle=FakeOracle(),
        db_path=str(BLUEPRINT_DB),
    )
    candidates = search.random_candidates(4)
    for candidate, score in zip(candidates, (0.8, 0.7, 0.2, 0.1)):
        candidate.observations.append(score)
    original = candidates[:2]
    search.last_portfolio = tuple(candidate.name for candidate in original)
    write_result(result_path, original)

    monkeypatch.setattr(
        search_module,
        "select_portfolio",
        lambda *args, **kwargs: (candidates[2:], 0.15, 0.5),
    )
    assert search.publish()
    assert json.loads(result_path.read_text()) == {
        "molecules": [candidate.name for candidate in original]
    }


def test_next_proposals_start_while_oracle_is_busy(tmp_path):
    cfg = challenge(num_molecules=20)

    proposals_started = Event()

    class CoordinatedOracle(FakeOracle):
        def score(self, targets, smiles):
            assert proposals_started.wait(timeout=5)
            return super().score(targets, smiles)

    search = Search(
        cfg,
        Settings(
            runtime_seconds=1_000,
            shutdown_guard_seconds=0,
            prediction_batch=24,
            proposal_multiplier=1,
            max_iterations=2,
        ),
        str(tmp_path / "result.json"),
        "/unused",
        oracle=CoordinatedOracle(),
        db_path=str(BLUEPRINT_DB),
    )
    original = search.discovery_proposals
    calls = 0

    def observed_proposals():
        nonlocal calls
        calls += 1
        if calls == 2:
            proposals_started.set()
        return original()

    search.discovery_proposals = observed_proposals
    search.run()
    assert calls == 2
    assert search.scorer.predictions == 48


def test_submission_tree_contains_no_binary_or_precomputed_artifacts():
    root = Path(__file__).resolve().parents[1]
    forbidden = {
        ".so",
        ".dylib",
        ".dll",
        ".pyc",
        ".pkl",
        ".pt",
        ".npy",
        ".npz",
        ".csv",
        ".json",
        ".sqlite",
        ".gz",
        ".zip",
    }
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(
            part in {".git", ".pixi", "__pycache__", ".pytest_cache"}
            for part in path.parts
        )
    ]
    assert not [path for path in files if path.suffix.lower() in forbidden]
