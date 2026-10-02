# Selector replay CPU validation

These are actual CPU software checks of the replaceable selector, static KV
cache, selected attention and strict probe request boundary. No pretrained
model, CUDA workload, physical probe, new dependency installation or benchmark
hardware rerun was performed.

## Final complete regression

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s artifacts/tests -v
~~~

**390 tests passed in 51.398 seconds**, exit 0, no skips.
All 288 prior tests remain. New coverage is 19 selector, 22 sparse attention,
21 probe boundary, 12 CLI routing, 15 CLI input-budget, 10 CLI output-budget
and 3 actual CPU integration tests.

The relevant source hashes were recorded before and after the complete suite
and remained equal:

| Source | SHA-256 |
| --- | --- |
| selector_replay.py | 7bcf69f0c45ba7a0ebf5f0ab1c4413f05d1264a82989afc5f0649375e5d836a0 |
| sparse_attention.py | 7b9e60bac5dbfb0ce8f69b6341708ed9ff5ed3f5373cef5f2c5ae31e049c0e7c |
| probe_boundary.py | e3845a7e3f086ab5561113633891b2d5fafbe853560624059d060a6131754d61 |
| selector_smoke.py | 6028c90af8561b53cb9b1104e7dfa6d9d5c31edb19c8e6a1788849cb31d6a233 |
| cli.py | 97d898440a6450612149a4be785ee6188a36ffe9c842bed18cce14899224873a |

Private log, relative to artifacts:
test-output/selector-writer-final-regression-20261002.log

Log SHA-256:
4af952842917ed59486dcb11e4d68d6ffebc7782e2fd762558eaf171e7ca7d61

Coverage includes ties/latest policy, plugin metadata and immutable snapshots,
causal absolute positions, sequence/identity/hash binding, strict GQA,
malformed/missing/duplicate/reordered inputs, finite values and overflow,
preallocation budgets, row-only cache writes, static pointers and forbidden
whole-cache copies. Unselected cache NaN/Inf leaves output unchanged; selected
nonfinite values fail. Probe requests refuse oracle fields at every nesting
level and retain only fresh primitive metadata.

Independent read-only reviews examined the selector contract and attention
math/GQA/read path. Logical tensor-path checks do not prove GPU cache-line
behavior or actual probe accuracy.

## Selector CLI input-budget repair

The post-repair selector CLI suite passed 27 tests in 0.193 seconds, exit 0.
The 15 new regressions use real tiny/sparse files and read/JSON-parse spies.
They verify rejection of a default-cap-plus-one 64 MiB sparse score or
manifest file before opening/parsing; a physical line exceeding 64 KiB after
reading only cap plus one; and a nonempty row beyond the config-derived count
before parsing that row. Blank lines consume bytes. Understated metadata,
descriptor budgets, bounded cumulative reads and file growth are covered.
Every read has an explicit size. Budget and strict JSON failures return
CLI exit 2 without creating a build output.

Parser-error tests explicitly inject ValueError and RecursionError instead
of assuming a fixed JSON depth always exceeds this environment's decoder
limit. Duplicate keys, nonfinite numbers, invalid UTF-8 and nonregular input
also fail. These source-byte caps do not bound Python heap usage to 64 MiB.

Actual post-repair selector-build and selector-validate both exited 0 over
the existing private synthetic config/score JSONL. The rebuilt manifest was
identical to the prior CPU smoke, including semantic digest
86e7d9e2cd548c1c4f5f10b9dedc9a94b01d06b6ee65bd1c246f6a7ec09d8e3d.

Private normal-command report, relative to artifacts:
test-output/selector-cli-p2-normal/cli-report.json

Report SHA-256:
aeb7565184d455e9f44adae9977da6cf14c26994680b9d80b76f432c3ec5b3d9

Independent review confirmed that the applied reader implementation matched
the reviewed proposal and preserved expected-config binding. The final
production CLI SHA is recorded above. Other CLI readers were not changed.

## Selector CLI output-budget consistency

The final complete CPU regression above was run on 2026-10-02. The source
hashes of the writer, budget tests and contract documentation were identical
before and after execution. The previously completed final targeted CLI suite
passed 37 tests in 0.246 seconds; all 37 are included in the 390-test suite.
The 10 output regressions use small parameterized caps and synthetic fixtures.

Real build/validate roundtrips cover the exact compact UTF-8 byte boundary,
including the trailing newline. The compact output fits where pretty output
would exceed the same cap. Reducing the cap by one rejects build with exit 2,
empty success output, no destination and no new parent directories. Tests
also forbid whole-string dumps and cover second-pass overflow, serialization,
short/write failures, stream-close/fdopen failures and preservation of user
replacement files. Replacement followed by continued complete encoding must
fail instead of reporting success for the wrong named output.

The writer counts iterencode UTF-8 chunks before any destination creation,
then writes them in a second binary pass with the same reader cap. It checks
actual path identity and path/descriptor sizes after the stream flush/close
while retaining the owned descriptor. Cleanup deletes only the writer's own
partial file when its inode still matches. Existing files are preserved.

Independent final read-only review confirmed the applied writer and tests,
including the previously pending success-path guard, and git diff --check
passed. No full-size multi-million-element fixture, pretrained model, GPU
workload or probe was run. This round performs only server-local validation
and commit; it does not upload, update Library or write GitHub.

## Persistent synthetic CPU closure

~~~bash
python -m artifacts.janus_artifact.cli selector-replay-smoke \
  --work-dir artifacts/test-output/selector-replay-final-smoke
~~~

Actual status ok, exit 0. Reported core interval 0.019427932798862457 s, measured
after Torch import and directory creation and before final report persistence.
It is not a model-performance benchmark.

| Actual software item | Result |
| --- | --- |
| Device / source | CPU / synthetic_tensor_fixture |
| Prompt / teacher-forced targets | 4 / 3 tokens |
| Layers / query heads / KV heads / head dimension | 2 / 4 / 2 / 3 |
| Selector | deterministic top-1 plus current legal token |
| Generation steps / layer-step evaluations | 3 / 6 |
| Query-head output comparisons | 24 |
| Float64 mathematical reference tolerance | rtol 1e-12, atol 1e-12 |
| Maximum absolute error observed | 0.0 |
| Cache K/V tensor storage | 1,152 bytes; pointers unchanged |
| Each GQA group union | strict subset of its valid causal prefix |
| Theoretical valid-prefix K/V bytes across evaluations | 2,880 |
| Theoretical gathered K/V bytes across evaluations | 1,488 |
| Probe input contracts validated | 3 |
| Actual model / probe / physical cache-line verification | false / false / false |

The two byte counts describe logical selected-row storage only. No hardware
traffic counter, cache-set measurement or physical pattern detection is claimed.
Synthetic reference evaluation is an oracle operation outside online attention.
Probe examples are host scheduling requests, not simulated latency observations.

The manifest was frozen before cache replay. Oracle token history, indices and
evaluation audits are stored in an independent private sidecar. Probe request
objects contain only schema, opaque run/step IDs, host timing and own-buffer
descriptors.

Actual CLI selector-build and selector-validate were also run over the
persisted synthetic config/JSONL scores. The rebuilt manifest was identical
to the original, including these semantic digests:

- Manifest: 86e7d9e2cd548c1c4f5f10b9dedc9a94b01d06b6ee65bd1c246f6a7ec09d8e3d
- Scores: e6ca395b3c140bacfabaa151946a095e2a6b7dff994f646f1322c61fe528623a
- Sequence: 9d9a57c3c2295b33ae0b61927d0aa6231c266328fd0399776590bfcf0f28e164

Private report, relative to artifacts:
test-output/selector-replay-final-smoke/report.json

Report and CLI result SHA-256:
36be40bd9bce0d3437e987bd7a94165496ee41aaac9446f47cf8c9b47da3a086

Runtime directories/files are private and excluded from source exports. Source
exports contain only code, tests, synthetic fixture sources and documentation.

## Remaining model and physical-probe gaps

This closure uses independent synthetic layer Q/K/V inputs. It does not run
LLaMA tokenizer, embeddings, RoPE, model forward, free generation or a trained
semantic attack. The cache/attention implementation currently supports FP32/64;
FP16/BF16 and an audited model/cache adapter remain future work.

Authorized real weights and exact revisions are pending. The existing PyTorch
environment suffices for this core; Transformers and Safetensors were not
installed. Physical cache/TLB observation, calibration and pattern observability
remain unimplemented. Probe output must be measured independently and joined
to oracle only after collection. Logical heads are not assumed physically
distinguishable.

The preserved prior owned-GPU benchmark has overall failed/cleanup unknown
after a post-query utilization of 13 percent and memory of 0 MiB. Its two owned
workers had exited and their allocators were empty; actual cross-context GPU
overlap remained unknown. This task neither reran it nor altered its outcome.

## Source and Library

The local selector branch starts from preserved source
91829d17d686ae6b9a7bf1fcb37b2f1f734000cc. Older branches and user files remain.
No GitHub write is performed. The complete new source-only snapshot is supplied
for review and replaces the same existing Library identity with a version guard;
the Library receipt reports the resulting version.
