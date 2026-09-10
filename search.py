"""Fast, source-only Boltz-2 search for the NOVA Blueprint sandbox.

The useful idea from our previous winner is preserved: expand rows and columns
around molecules that scored well as complete molecules. Nothing is loaded
from an earlier run; every observation and every model fit is created here.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sqlite3
import time
from typing import Iterable, Protocol, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor

import nova_miner
from nova_miner.utils.molecules import get_smiles
from nova_miner.utils.oracle import Oracle, combine

from portfolio import select_portfolio, write_result

log = logging.getLogger("nova-boltz2-miner")
RDLogger.DisableLog("rdApp.*")
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@dataclass(frozen=True)
class Challenge:
    targets: tuple[str, ...]
    antitargets: tuple[str, ...]
    antitarget_weight: float
    num_molecules: int
    min_heavy_atoms: int
    min_rotatable_bonds: int
    max_rotatable_bonds: int
    entropy_threshold: float
    tanimoto_threshold: float
    allowed_reaction: int | None

    @property
    def proteins(self) -> tuple[str, ...]:
        return self.targets + self.antitargets

    @property
    def rng_seed(self) -> int:
        # Reproducible exploration derived only from the current challenge.
        digest = hashlib.sha256("|".join(self.proteins).encode("ascii")).digest()
        return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class Settings:
    runtime_seconds: float = 3_600.0
    shutdown_guard_seconds: float = 15.0
    discovery_fraction: float = 0.84
    prediction_batch: int = 96
    proposal_multiplier: int = 8
    min_surrogate_samples: int = 96
    surrogate_refit_every: int = 3
    entropy_margin: float = 0.006
    oracle_timeout_seconds: float = 480.0
    max_iterations: int | None = None

    @classmethod
    def from_environment(cls) -> "Settings":
        """Production defaults; overrides make short sandbox tests possible."""
        maximum = os.environ.get("NOVA_MAX_ITERATIONS")
        return cls(
            runtime_seconds=float(os.environ.get("NOVA_RUNTIME_SECONDS", "3600")),
            shutdown_guard_seconds=float(
                os.environ.get("NOVA_SHUTDOWN_GUARD_SECONDS", "15")
            ),
            discovery_fraction=float(os.environ.get("NOVA_DISCOVERY_FRACTION", "0.84")),
            prediction_batch=int(os.environ.get("NOVA_PREDICTION_BATCH", "96")),
            proposal_multiplier=int(os.environ.get("NOVA_PROPOSAL_MULTIPLIER", "8")),
            min_surrogate_samples=int(
                os.environ.get("NOVA_MIN_SURROGATE_SAMPLES", "96")
            ),
            surrogate_refit_every=int(
                os.environ.get("NOVA_SURROGATE_REFIT_EVERY", "3")
            ),
            entropy_margin=float(os.environ.get("NOVA_ENTROPY_MARGIN", "0.006")),
            oracle_timeout_seconds=float(
                os.environ.get("NOVA_ORACLE_TIMEOUT_SECONDS", "480")
            ),
            max_iterations=int(maximum) if maximum else None,
        )


def load_challenge(path: str) -> Challenge:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = {**payload.get("config", {}), **payload.get("challenge", {})}
    allowed = values.get("allowed_reaction")
    if isinstance(allowed, str) and allowed.startswith("rxn:"):
        allowed = allowed.split(":", 1)[1]
    challenge = Challenge(
        targets=tuple(values["target_sequences"]),
        antitargets=tuple(values.get("antitarget_sequences", [])),
        antitarget_weight=float(values["antitarget_weight"]),
        num_molecules=int(values["num_molecules"]),
        min_heavy_atoms=int(values["min_heavy_atoms"]),
        min_rotatable_bonds=int(values["min_rotatable_bonds"]),
        max_rotatable_bonds=int(values["max_rotatable_bonds"]),
        entropy_threshold=float(values["entropy_min_threshold"]),
        tanimoto_threshold=float(values.get("tanimoto_max_threshold", 1.0)),
        allowed_reaction=int(allowed) if allowed not in (None, "") else None,
    )
    if not challenge.targets or not challenge.proteins:
        raise ValueError("challenge has no target")
    return challenge


@dataclass(frozen=True)
class Reaction:
    reaction_id: int
    component_pools: tuple[tuple[int, ...], ...]


@dataclass(eq=False)
class Candidate:
    name: str
    smiles: str
    inchikey: str
    heavy_atoms: int
    reaction_id: int
    components: tuple[int, ...]
    morgan: object
    maccs: np.ndarray
    observations: list[float] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.observations)) if self.observations else -math.inf

    def conservative_score(self, noise_floor: float) -> float:
        """Penalize one-off wins and noisy repeat scores."""
        if not self.observations:
            return -math.inf
        if len(self.observations) == 1:
            return self.observations[0] - noise_floor
        deviation = float(np.std(self.observations, ddof=1))
        error = max(noise_floor, deviation) / math.sqrt(len(self.observations))
        return self.mean_score - 0.45 * error


def parse_name(name: str) -> tuple[int, tuple[int, ...]]:
    parts = str(name).split(":")
    if len(parts) not in (4, 5) or parts[0] != "rxn":
        raise ValueError(name)
    return int(parts[1]), tuple(int(value) for value in parts[2:])


def reaction_name(reaction_id: int, components: Sequence[int]) -> str:
    fields = ["rxn", str(reaction_id), *(str(component) for component in components)]
    return ":".join(fields)


class ReactionSpace:
    """Load component IDs once, then propose tuples without database traffic."""

    def __init__(self, db_path: str, allowed_reaction: int | None):
        self.db_path = str(Path(db_path).resolve())
        self.reactions = self._load(allowed_reaction)
        self.by_id = {reaction.reaction_id: reaction for reaction in self.reactions}
        if not self.reactions:
            raise ValueError("no eligible reactions")

    def _connect(self):
        return sqlite3.connect(f"file:{self.db_path}?mode=ro&immutable=1", uri=True)

    def _load(self, allowed_reaction: int | None) -> tuple[Reaction, ...]:
        reactions = []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT CAST(rxn_id AS INT), roleA, roleB, roleC "
                "FROM reactions ORDER BY CAST(rxn_id AS INT)"
            ).fetchall()
            for reaction_id, role_a, role_b, role_c in rows:
                if allowed_reaction is not None and reaction_id != allowed_reaction:
                    continue
                masks = [role_a, role_b] + ([role_c] if role_c else [])
                pools = []
                for mask in masks:
                    pool = tuple(
                        row[0]
                        for row in connection.execute(
                            "SELECT mol_id FROM molecules WHERE (role_mask & ?) = ? ORDER BY mol_id",
                            (mask, mask),
                        )
                    )
                    if not pool:
                        break
                    pools.append(pool)
                if len(pools) == len(masks):
                    reactions.append(Reaction(reaction_id, tuple(pools)))
        return tuple(reactions)

    def random_names(self, rng: random.Random, count: int):
        order = list(self.reactions)
        rng.shuffle(order)
        for index in range(count):
            reaction = order[index % len(order)]
            components = [rng.choice(pool) for pool in reaction.component_pools]
            yield reaction_name(reaction.reaction_id, components)

    def neighbour_names(
        self, rng: random.Random, seeds: Sequence[Candidate], count: int
    ):
        """Vary one axis around complete molecules that the oracle liked."""
        if not seeds:
            yield from self.random_names(rng, count)
            return
        seeds = list(seeds)
        rng.shuffle(seeds)
        for index in range(count):
            seed = seeds[index % len(seeds)]
            reaction = self.by_id[seed.reaction_id]
            components = list(seed.components)
            axis = (index // len(seeds)) % len(components)
            components[axis] = rng.choice(reaction.component_pools[axis])
            yield reaction_name(reaction.reaction_id, components)


class CandidateStore:
    """Materialize each tuple once and retain only validator-valid molecules."""

    def __init__(self, challenge: Challenge):
        self.challenge = challenge
        self.candidates: dict[str, Candidate] = {}
        self.inchikeys: set[str] = set()
        self.attempted: set[str] = set()

    def materialize(self, names: Iterable[str], limit: int) -> list[Candidate]:
        accepted = []
        for name in names:
            if len(accepted) >= limit:
                break
            if name in self.attempted:
                continue
            self.attempted.add(name)
            try:
                reaction_id, components = parse_name(name)
                if (
                    self.challenge.allowed_reaction is not None
                    and reaction_id != self.challenge.allowed_reaction
                ):
                    continue
                smiles = get_smiles(name)
                molecule = Chem.MolFromSmiles(smiles) if smiles else None
                if molecule is None:
                    continue
                heavy_atoms = molecule.GetNumHeavyAtoms()
                rotatable = Descriptors.NumRotatableBonds(molecule)
                if heavy_atoms < self.challenge.min_heavy_atoms:
                    continue
                if not (
                    self.challenge.min_rotatable_bonds
                    <= rotatable
                    <= self.challenge.max_rotatable_bonds
                ):
                    continue
                inchikey = Chem.MolToInchiKey(molecule)
                if not inchikey or inchikey in self.inchikeys:
                    continue
                candidate = Candidate(
                    name=name,
                    smiles=smiles,
                    inchikey=inchikey,
                    heavy_atoms=heavy_atoms,
                    reaction_id=reaction_id,
                    components=components,
                    morgan=MORGAN.GetFingerprint(molecule),
                    maccs=np.asarray(MACCSkeys.GenMACCSKeys(molecule), dtype=np.uint8),
                )
            except Exception:
                continue
            self.candidates[name] = candidate
            self.inchikeys.add(inchikey)
            accepted.append(candidate)
        return accepted

    def scored(self) -> list[Candidate]:
        return [
            candidate
            for candidate in self.candidates.values()
            if candidate.observations
        ]

    def noise_floor(self) -> float:
        repeat_noise = [
            float(np.std(candidate.observations, ddof=1))
            for candidate in self.candidates.values()
            if len(candidate.observations) >= 2
        ]
        return max(0.003, float(np.median(repeat_noise)) if repeat_noise else 0.0)

    def elite_seeds(self, count: int) -> list[Candidate]:
        """Pick high scorers that do not reuse components within a reaction."""
        noise = self.noise_floor()
        ranked = sorted(
            self.scored(), key=lambda item: item.conservative_score(noise), reverse=True
        )
        selected = []
        used: dict[int, list[set[int]]] = {}
        for candidate in ranked:
            roles = used.setdefault(
                candidate.reaction_id, [set() for _ in candidate.components]
            )
            if any(
                component in roles[index]
                for index, component in enumerate(candidate.components)
            ):
                continue
            selected.append(candidate)
            for index, component in enumerate(candidate.components):
                roles[index].add(component)
            if len(selected) == count:
                break
        return selected


class OracleLike(Protocol):
    def score(self, targets: list[str], smiles: list[str]) -> list[dict]: ...


class OracleScorer:
    def __init__(
        self,
        challenge: Challenge,
        settings: Settings,
        socket_path: str,
        oracle: OracleLike | None = None,
    ):
        self.challenge = challenge
        self.prediction_batch = max(len(challenge.proteins), settings.prediction_batch)
        self.oracle = oracle or Oracle(
            socket_path, timeout=settings.oracle_timeout_seconds
        )
        self.durations: list[float] = []
        self.predictions = 0
        self.last_scores: list[float] = []

    @property
    def molecules_per_request(self) -> int:
        return max(1, self.prediction_batch // len(self.challenge.proteins))

    @property
    def expected_seconds(self) -> float:
        recent = self.durations[-3:]
        return max(20.0, sum(recent) / len(recent)) if recent else 230.0

    def score(self, candidates: Sequence[Candidate]) -> tuple[int, float]:
        batch = list(candidates[: self.molecules_per_request])
        self.last_scores = []
        started = time.monotonic()
        rows = self.oracle.score(
            list(self.challenge.proteins), [candidate.smiles for candidate in batch]
        )
        elapsed = time.monotonic() - started
        self.durations.append(elapsed)

        accepted = 0
        for candidate, row in zip(batch, rows):
            if not isinstance(row, dict) or row.get("smiles") != candidate.smiles:
                continue
            metrics = row.get("scores")
            if not isinstance(metrics, list) or len(metrics) != len(
                self.challenge.proteins
            ):
                continue
            values = [combine(metric, candidate.heavy_atoms) for metric in metrics]
            if not all(math.isfinite(value) for value in values):
                continue
            target_count = len(self.challenge.targets)
            target_score = sum(values[:target_count]) / target_count
            anti = values[target_count:]
            antitarget_score = sum(anti) / len(anti) if anti else 0.0
            score = target_score - self.challenge.antitarget_weight * antitarget_score
            candidate.observations.append(score)
            self.last_scores.append(score)
            accepted += 1
        self.predictions += len(batch) * len(self.challenge.proteins)
        return accepted, elapsed


def fingerprint_matrix(candidates: Sequence[Candidate]) -> np.ndarray:
    matrix = np.zeros((len(candidates), 2049), dtype=np.float32)
    for index, candidate in enumerate(candidates):
        on_bits = list(candidate.morgan.GetOnBits())
        matrix[index, on_bits] = 1.0
        matrix[index, 2048] = candidate.heavy_atoms / 100.0
    return matrix


class LiveSurrogate:
    """ExtraTrees trained only on current-run oracle observations."""

    def __init__(self, rng_seed: int):
        self.model = ExtraTreesRegressor(
            n_estimators=56,
            max_depth=18,
            min_samples_leaf=2,
            max_features=0.35,
            n_jobs=-1,
            random_state=rng_seed & 0x7FFFFFFF,
        )
        self.fitted = False

    def fit(self, candidates: Sequence[Candidate]) -> None:
        training = [candidate for candidate in candidates if candidate.observations]
        self.model.fit(
            fingerprint_matrix(training),
            [candidate.mean_score for candidate in training],
        )
        self.fitted = True

    def choose(
        self,
        proposals: Sequence[Candidate],
        count: int,
        rng: np.random.Generator,
    ) -> list[Candidate]:
        proposals = list(proposals)
        if not self.fitted or len(proposals) <= count:
            return proposals[:count]
        features = fingerprint_matrix(proposals)
        tree_scores = np.asarray(
            [
                tree.predict(features, check_input=False)
                for tree in self.model.estimators_
            ]
        )
        mean = tree_scores.mean(axis=0)
        uncertainty = tree_scores.std(axis=0)

        exploit_count = int(count * 0.65)
        uncertain_count = int(count * 0.25)
        chosen: list[int] = []
        used: set[int] = set()

        for index in np.argsort(mean)[::-1]:
            if len(chosen) == exploit_count:
                break
            chosen.append(int(index))
            used.add(int(index))
        for index in np.argsort(mean + 0.8 * uncertainty)[::-1]:
            if len(chosen) == exploit_count + uncertain_count:
                break
            if int(index) not in used:
                chosen.append(int(index))
                used.add(int(index))
        remaining = np.asarray(
            [index for index in range(len(proposals)) if index not in used], dtype=int
        )
        random_count = min(count - len(chosen), len(remaining))
        if random_count:
            chosen.extend(
                int(index)
                for index in rng.choice(remaining, size=random_count, replace=False)
            )
        return [proposals[index] for index in chosen]


class Search:
    def __init__(
        self,
        challenge: Challenge,
        settings: Settings,
        output_path: str,
        oracle_socket: str,
        oracle: OracleLike | None = None,
        db_path: str | None = None,
    ):
        database = db_path or str(
            Path(nova_miner.__file__).resolve().parent
            / "combinatorial_db"
            / "molecules.sqlite"
        )
        self.challenge = challenge
        self.settings = settings
        self.output_path = output_path
        self.space = ReactionSpace(database, challenge.allowed_reaction)
        self.store = CandidateStore(challenge)
        self.scorer = OracleScorer(challenge, settings, oracle_socket, oracle)
        self.random = random.Random(challenge.rng_seed)
        self.numpy_random = np.random.default_rng(challenge.rng_seed)
        self.surrogate = LiveSurrogate(challenge.rng_seed)
        self.last_portfolio: tuple[str, ...] | None = None

    def random_candidates(self, count: int) -> list[Candidate]:
        names = self.space.random_names(self.random, max(512, count * 8))
        return self.store.materialize(names, count)

    def discovery_proposals(self) -> list[Candidate]:
        count = self.scorer.molecules_per_request
        proposal_count = count * self.settings.proposal_multiplier
        seeds = self.store.elite_seeds(18)
        neighbour_count = int(proposal_count * 0.82) if seeds else 0
        names = list(
            self.space.neighbour_names(self.random, seeds, neighbour_count)
        ) + list(self.space.random_names(self.random, proposal_count - neighbour_count))
        proposals = self.store.materialize(names, proposal_count)
        if len(proposals) < count:
            proposals.extend(self.random_candidates(count - len(proposals)))
        return proposals

    def discovery_batch(
        self, proposals: Sequence[Candidate] | None = None
    ) -> list[Candidate]:
        proposals = (
            list(proposals) if proposals is not None else self.discovery_proposals()
        )
        return self.surrogate.choose(
            proposals, self.scorer.molecules_per_request, self.numpy_random
        )

    def repeat_batch(self) -> list[Candidate]:
        count = self.scorer.molecules_per_request
        noise = self.store.noise_floor()
        leaders = sorted(
            self.store.scored(),
            key=lambda item: item.conservative_score(noise),
            reverse=True,
        )[: max(count, self.challenge.num_molecules * 2)]
        leaders.sort(
            key=lambda item: (len(item.observations), -item.conservative_score(noise))
        )
        return leaders[:count]

    def publish(self, include_unscored: bool = False) -> bool:
        candidates = (
            list(self.store.candidates.values())
            if include_unscored
            else self.store.scored()
        )
        noise_floor = self.store.noise_floor()
        portfolio, score, entropy = select_portfolio(
            candidates,
            self.challenge.num_molecules,
            self.challenge.tanimoto_threshold,
            self.challenge.entropy_threshold + self.settings.entropy_margin,
            noise_floor,
        )
        if len(portfolio) != self.challenge.num_molecules:
            return False
        if entropy <= self.challenge.entropy_threshold:
            return False
        names = tuple(candidate.name for candidate in portfolio)
        if names == self.last_portfolio:
            return True
        if self.last_portfolio is not None:
            previous = [
                self.store.candidates[name] for name in self.last_portfolio
            ]
            previous_score = float(
                np.mean(
                    [
                        candidate.conservative_score(noise_floor)
                        for candidate in previous
                    ]
                )
            )
            if score <= previous_score:
                return True
        write_result(self.output_path, portfolio)
        self.last_portfolio = names
        log.info(
            "checkpoint molecules=%d robust_mean=%.6f entropy=%.5f scored=%d predictions=%d",
            len(portfolio),
            score,
            entropy,
            len(self.store.scored()),
            self.scorer.predictions,
        )
        return True

    def bootstrap(self) -> None:
        target = self.challenge.num_molecules * 3
        for _ in range(5):
            self.random_candidates(target - len(self.store.candidates))
            if self.publish(include_unscored=True):
                return
            target += self.challenge.num_molecules
        raise RuntimeError("could not build a validator-safe initial portfolio")

    def run(self) -> None:
        started = time.monotonic()
        deadline = started + self.settings.runtime_seconds
        discovery_deadline = (
            started + self.settings.runtime_seconds * self.settings.discovery_fraction
        )
        self.bootstrap()

        iteration = 0
        prepared_proposals = None
        # Construct the next candidate pool while the current oracle request
        # runs. The one-batch delay also keeps exploration from collapsing too
        # quickly around the newest noisy winners.
        with ThreadPoolExecutor(max_workers=1) as proposal_worker:
            while True:
                remaining = deadline - time.monotonic()
                safe_request_time = (
                    self.scorer.expected_seconds + self.settings.shutdown_guard_seconds
                )
                if remaining <= safe_request_time:
                    log.info(
                        "stopping before unsafe oracle request; %.1fs remain", remaining
                    )
                    break
                if (
                    self.settings.max_iterations is not None
                    and iteration >= self.settings.max_iterations
                ):
                    break
                iteration += 1
                discovery = time.monotonic() < discovery_deadline
                if discovery:
                    batch = self.discovery_batch(prepared_proposals)
                    has_next_iteration = (
                        self.settings.max_iterations is None
                        or iteration < self.settings.max_iterations
                    )
                    next_proposals = (
                        proposal_worker.submit(self.discovery_proposals)
                        if has_next_iteration
                        else None
                    )
                else:
                    batch = self.repeat_batch()
                    next_proposals = None
                if not batch:
                    break
                try:
                    accepted, elapsed = self.scorer.score(batch)
                except Exception as error:
                    log.warning(
                        "oracle request failed; previous checkpoint preserved: %s",
                        error,
                    )
                    accepted, elapsed = 0, 0.0

                # Join before publishing so no thread mutates the candidate store
                # while the portfolio is being assembled.
                prepared_proposals = (
                    next_proposals.result() if next_proposals is not None else None
                )
                log.info(
                    "iteration=%d phase=%s molecules=%d accepted=%d oracle=%.1fs "
                    "batch_mean=%.6f batch_p90=%.6f batch_best=%.6f remaining=%.1fs",
                    iteration,
                    "discovery" if discovery else "repeat",
                    len(batch),
                    accepted,
                    elapsed,
                    float(np.mean(self.scorer.last_scores))
                    if self.scorer.last_scores
                    else -math.inf,
                    float(np.quantile(self.scorer.last_scores, 0.9))
                    if self.scorer.last_scores
                    else -math.inf,
                    max(self.scorer.last_scores, default=-math.inf),
                    deadline - time.monotonic(),
                )
                if not accepted:
                    time.sleep(1.0)
                    continue
                if (
                    discovery
                    and len(self.store.scored()) >= self.settings.min_surrogate_samples
                    and (
                        not self.surrogate.fitted
                        or iteration % self.settings.surrogate_refit_every == 0
                    )
                ):
                    fit_started = time.monotonic()
                    self.surrogate.fit(self.store.scored())
                    log.info(
                        "surrogate fit samples=%d time=%.2fs",
                        len(self.store.scored()),
                        time.monotonic() - fit_started,
                    )
                self.publish()
        self.publish()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    input_path = os.environ.get("NOVA_INPUT", "/workspace/input.json")
    output_path = os.path.join(os.environ.get("OUTPUT_DIR", "/output"), "result.json")
    challenge = load_challenge(input_path)
    settings = Settings.from_environment()
    log.info(
        "starting proteins=%d output=%d prediction_batch=%d runtime=%.0fs",
        len(challenge.proteins),
        challenge.num_molecules,
        settings.prediction_batch,
        settings.runtime_seconds,
    )
    Search(
        challenge,
        settings,
        output_path,
        os.environ["ORACLE_SOCKET"],
    ).run()
