# Isolated Nexto Demonstration Collector (Stage B)

## Overview
This directory contains the isolated demonstration-collection harness for research into teacher-to-student policy distillation under the Model Wars competition rules.

All code, tests, logs, and configurations in this directory are strictly isolated from the production submission files.

---

## Directory Layout
```text
scratch/distillation/
├── README.md                     # This documentation file
├── collector.py                  # Isolated collector, action mapper & teacher harness
├── collector_config.json         # Configuration parameters and safety guards
├── COMPATIBILITY_RESULTS.md      # Empirical smoke test and unit test results
└── tests/
    ├── test_action_mapper.py     # Unit tests for 90-action classification
    ├── test_schema_validator.py  # Unit tests for 13D schema and split purity
    ├── test_sequence_alignment.py# Unit tests for 64-tick sequence chunking
    └── test_teacher_smoke.py     # Unit tests for Nexto loading & forward pass
```

---

## Safety and Competition Boundaries
1. **Zero Production Mutation:** `model/V13.pt`, `src/runtime.py`, `src/policy.py`, and `submission.json` are **never** imported, modified, or overwritten by this harness.
2. **Fixed Architecture Integrity:** The collector enforces the student's strict 13-feature input vector and 26,181 parameter budget.
3. **Action Filtering:** Nexto actions that are physically unrepresentable by the student decoder (reverse throttle, handbrake powerslide, aerial flight) are automatically filtered and masked out to prevent corrupting the student policy.
4. **Environment Isolation:** Nexto dependencies (`rlgym_compat`, `numpy==2.*`) are kept separate from the participant environment (`numpy==1.26.4`).

---

## How to Run Tests

### 1. Run Teacher Smoke Test & Action Audit
```powershell
python scratch/distillation/collector.py
```

### 2. Run Automated Unit Test Suite
```powershell
python -m unittest discover -s scratch/distillation/tests -p "test_*.py" -v
```

---

## Status and Readiness
- **Stage B Status:** Complete and verified (18/18 unit tests passed).
- **Prerequisite for Stage C:** Construction of an isolated virtual environment containing `rlgym_compat` to enable live in-game packet ingestion by Nexto during match play.
