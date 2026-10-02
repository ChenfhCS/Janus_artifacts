# Controlled upstream software validation

These are actual CPU software checks against generated toy workloads.
Simulated probes and oracle-aligned phase/gold annotations are explicit;
scientific_result and paper_reproduction are false.

## Complete regression

Command, from the repository:

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s artifacts/tests -v
~~~

Result after the P2 timestamp precision repair: **257 tests passed
in 52.481 seconds** (exit 0).
This includes all 171 prior offline/QAI/ATR tests and 86 upstream tests:
25 probe-contract, 24 reconstruction, 19 workload, 15 bridge and 3 CLI tests.
No tests were skipped.

Environment: Python 3.12.3, NumPy 2.3.2,
PyTorch 2.8.0+cu128; new upstream training runs explicitly on CPU.

Preserved local evidence, relative to artifacts:
- test-output/upstream-time-precision-regression.log
- Log SHA-256: 36696f66fa44936a96118acd25b99eb7455ad1090a0b8e1ffd1a2533a6add756

Coverage includes the original audit fixes (training provenance/payload split
isolation, checkpoint attributes and finite/contract validation, singleton-tail
batch handling and bounded NPZ/NPY loading), ATR ID/alignment/denominators, plus
raw/oracle separation, mapping/calibration/epoch consistency, allocation budgets,
raw simulator evidence, frozen splits, strict threshold equality, noise votes,
phase skew/ties/collapse, mass conservation, local density, numeric overflow,
gold independence, lossy QAI adaptation and output-preservation failures.

## P2 timestamp precision audit repair

The previous mixed int/float arithmetic incorrectly accepted a 100 ns alignment
error at epoch B = 2**60, reporting zero under both zero and default 5 ns
tolerance. An actual before/after synthetic check now rejects both settings,
and a 100 ns tolerance accepts with reported integer error 100 and exact
integer clock offset B.

Eight additional CPU tests cover the supplied counterexample, mixed integral
float/integer epoch translation, true fractional ties with earlier/later/reject
policies and nextafter tolerance, exact collection-end coverage, fractional
offsets and already-parsed IEEE values, unrepresentable offset/error rejection,
and a 0.25 ns error surviving a large epoch offset. All original tests remain.

Clock subtraction, boundary/end addition, distance, tie and tolerance decisions
use exact rational arithmetic. Existing numeric diagnostics are emitted only
when an int64 integer or finite float represents the exact result; unsupported
precision/range is rejected. Float inputs mean their existing IEEE binary
values, not recovered decimal text or precision lost before ingestion. See
UPSTREAM_RECONSTRUCTION.md for the complete supported-time contract.

Evidence, relative to artifacts:
- test-output/probe-time-precision-repro-before.json
- Before SHA-256: 9b564ae8f3c88d30093d8ca653c7193ae7c3608cd25b0b34a60a981d8816f85a
- test-output/probe-time-precision-repro-after.json
- After SHA-256: 51d35b51a05b677e9d4286b8bcd3a85c512c3e4946a9c6e68e02bf0437b232cb

The new full regression also runs the existing connected CPU bridge
training/gradient/update/safe-reload check. The persistent report below is
preserved evidence from source commit
695d07dc75439e60e5ef69b8f19a8a2a64d14f15; its duration and metric values are
not claimed as a newly rerun persistent report after this patch.

## Persistent connected smoke (source 695d07d)

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m artifacts.janus_artifact.cli upstream-synthetic-smoke \
  --work-dir artifacts/test-output/upstream-final-smoke --device cpu
~~~

Result: status ok, duration 8.975917 seconds.
The new output directory was checked absent before execution; older outputs
and the original local branch were preserved.

| Actual item | Result |
| --- | --- |
| Controlled cases / runs | 24 |
| Train / validation / test response groups | 18 / 3 / 3 |
| Simulated raw observations | 29,184 |
| Reconstructed trace / replay records | 96 / 96 |
| QAI queries / ATR responses / ATR steps | 24 / 24 / 72 |
| ATR train-only candidates | [11, 17] |
| QAI train samples / validation samples | 18 / 3 |
| ATR train steps / validation steps | 54 / 9 |
| Model input (both pipelines) | [8, 4, 4] |
| Model parameters (each) | 12,014 |
| Epochs / batch / base channels (each) | 1 / 4 / 2 |
| QAI train loss / ATR train loss | 0.740567376216 / 0.748779859808 |
| Real gradient / parameter update (both) | true / true |
| Safe checkpoint reload (both) | verified using existing weights_only=True loaders |
| QAI test predictions / PASR denominator | 3 / 3 |
| ATR test predictions / DASR denominator | 9 / 9 |
| OOV gold tokens counted incorrect | 3 |
| Missing / extra ATR predictions | 0 / 0 |

The smoke PASR is 0/3
(0.000000); DASR is 6/9
and macro 0.666667. These values test connected execution and complete
denominators; they are not attack efficacy or paper-reproduction results.
The test-only zero-sum token 23 remains outside train candidates [11, 17].

Preserved local evidence:
- test-output/upstream-final-smoke/report.json
- Report SHA-256: b407527b375e72c495c42d0a23e29b435740cd520f3da5c9fde9345489a28476
- Bundle SHA-256: 4267cb00fc1b6af89e88319904c1b35a77e12b86b9d82e4407c71e3745457956
- Frozen assignments SHA-256: 5183609f1ffa48b94ed350a5e9dab07026e9eb881b2fa7c5c62294e1ec8bdc01

The report links generated raw/profiles/traces/replay/manifests/mapping,
predictions and safe checkpoints in that runtime directory. Runtime payloads,
predictions, checkpoints and caches are excluded from the source-only export.

## Interpretation and remaining evidence

Reference alignment and all-step vote grouping are declared choices, without
autonomous trace-only detection or cross-step sliding windows. Representative
power-law and density/ceil/prefix profiles are not recovered attention.
The QAI uint16 rank surrogate is a lossy adapter with per-run normalization
and two selected memberships. See UPSTREAM_RECONSTRUCTION.md for exact meaning.

Physical probe collection/calibration, author-matched reconstruction choices,
complete real train/test records and paper attack-rate evaluation remain absent.
No legacy attack scripts, unknown pickle/PT or external targets were executed.
