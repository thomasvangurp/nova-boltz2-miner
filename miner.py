"""NOVA Blueprint Boltz-2 miner entry point.

The validator runs this file as ``python /workspace/miner.py``.  All search
state is learned from the current challenge and its oracle responses.
"""

from search import main


if __name__ == "__main__":
    main()
