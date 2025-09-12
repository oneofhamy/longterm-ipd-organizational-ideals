# Generational IPD Simulation: Shadow Cabals, Organizational Memory & Institutional Drift

**Author:** David A. Cook 
**License:** AGPL-3.0

## What is this?

This repository contains a multi-agent simulation of the Iterated Prisoner’s Dilemma (IPD) with generational turnover, living/organizational memory, and emergent "shadow cabal" detection—modeling how institutions, founding ideals, and legacy power structures persist, mutate, and drift over time.

## Features

- **Agent-based IPD:** Agents play repeated Prisoner's Dilemma with realistic strategies (TFT, MoQ, GTFT, Ethnocentric, etc.).
- **Generational Turnover:** Agents age, die, and are replaced by “children” inheriting traits.
- **Memory Decay & Perception:** Agents forget old partners, with karma and reputation perception decoupled from true behavior.
- **Shadow Cabal Detection:** Detects clusters (“cabals”) whose influence persists across generations even after direct memory is lost.
- **Founding Ideals:** Models trauma, martyrdom, and betrayal as events that encode organizational "founding myths."
- **Organizational Drift:** Group/agent ideals drift, imprint, and sometimes mythologize, producing highly realistic institutional evolution.
- **Comprehensive Analytics:** Full logging and post-simulation analytics/plots (karma drift, cluster power, shadow cabals, manipulators).

## Why does this matter?

Most ABMs ignore long-term institutional memory, shadow influence, and generational myth-drift. This project is for researchers, simulation nerds, political scientists, and anyone interested in how real-world organizations and “cults” keep their hidden power.

## How to run

1. `pip install -r requirements.txt`
2. Run `simulation.py` for a CLI output or `analytics.ipynb` for interactive exploration and plots.

## Example Plots


## License

AGPL-3.0

---

*Questions or collaboration? Open an issue or submit a PR!*
