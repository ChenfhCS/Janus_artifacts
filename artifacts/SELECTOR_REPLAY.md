# Replaceable selector replay core

This is a CPU-testable software core for offline selection, static KV storage
and attention over selected rows. It loads no pretrained model and implements
no physical probe. The next model adapter will consume the same manifest;
a score-based top-k strategy is the first implementation, not an exact
reproduction of a paper selector.

## Module and command boundary

- selector_replay.py: replaceable TokenSelector protocol, deterministic offline
  score selection, complete manifest validation and expected-config binding.
- sparse_attention.py: preallocated append-only KV tensors and GQA attention
  over selected cache rows.
- selector_smoke.py: short synthetic CPU layer inputs, reference evaluation and
  an independent oracle sidecar.
- probe_boundary.py: strict request metadata for a future independent probe.
  It performs no allocation or execution.

~~~bash
python -m artifacts.janus_artifact.cli selector-build \
  --config selector-config.json --scores offline-scores.jsonl \
  --output new-selector-manifest.json

python -m artifacts.janus_artifact.cli selector-validate \
  --manifest new-selector-manifest.json --expected-config selector-config.json

python -m artifacts.janus_artifact.cli selector-replay-smoke \
  --work-dir artifacts/test-output/selector-replay-new
~~~

Build output must be a new file; smoke work_dir must be new. Existing files,
links and permissions are preserved. The smoke command runs on CPU.
Only the existing standard library and PyTorch are needed for this core.

## Selector CLI input and output budgets

The selector commands use their own bounded binary readers. Score JSONL has a
64 MiB total raw-byte budget and a 64 KiB physical-line budget, including the
line ending. Blank lines consume both byte budgets. A line is bounded before
UTF-8 decoding and JSON parsing. At most G * layers * query_heads nonempty
score rows are parsed; the next nonempty row is refused before its JSON is
parsed. The core separately requires the exact complete set of causal rows.

Manifest JSON has a 64 MiB raw-byte budget before decoding or parsing. Regular
file and descriptor size checks, bounded reads and stability checks prevent
oversized input or file growth from bypassing the read budget. Config retains
its existing 1 MiB input budget. These are source-byte limits, not a claim that
Python's parsed objects use exactly that much heap memory. Oversized input
returns CLI exit 2 before a new output is created. Other CLI readers are
unchanged.

Build writes compact UTF-8 JSON with sorted keys, no indentation, and a final
newline. It uses the same 64 MiB manifest-byte cap as validate. Before making
new directories or a destination file, the writer traverses the JSON encoder
and counts encoded bytes, including the newline. It does not assemble a full
serialized string or a list of encoding chunks. A second binary streaming
pass writes only after that check and enforces the cap again. On serialization
or write failure, cleanup removes only the writer's own partial destination
if its inode still matches; existing files and replacement paths are preserved.

CLI support requires both the core config/score contract and a serialized
manifest that fits the shared reader/writer cap. Build never reports success
for an output above that cap. JSON formatting does not change the semantic
manifest digest.

## Fixed sequence and interchangeable strategy

SelectorReplayConfig persists all 19 fields explicitly: opaque run ID;
model/tokenizer identity and revision; model config and optional weight digests;
source kind; seed; prompt and teacher-forced token IDs; layer/query-head/KV-head
counts and head dimension; top_k; latest-token policy; selector name/version.
The synthetic source permits a null weight digest. An offline model-score
source requires a declared weight digest. This module does not inspect weights
or authenticate that declaration; a future model adapter must verify bytes.

The manifest fixes the entire short teacher-forced sequence before replay.
For P prompt tokens and G forced targets, decoding step s (zero-based) uses:

| Field | Exact meaning |
| --- | --- |
| query_position | P + s - 1, an absolute token position |
| cache_length | P + s |
| history | prompt + forced[:s] |
| target_token_id | forced[s], predicted at this step |
| static cache capacity | P + G - 1 |

The final forced target is predicted and is not inserted into the replay
cache. This core handles decoding queries; it does not define separate
per-query prefill selection or a complete model generation loop.

Each offline score row identifies step, layer and query head and contains
exactly cache_length finite builtin numbers. Duplicate/missing rows and future
score slots are refused. Input row order is canonicalized.

DeterministicTopKSelector ranks descending scores and breaks ties by ascending
absolute KV position. It selects min(top_k, cache_length), then the builder
optionally adds the current legal query position. The stored positions are
strictly ascending and unique; the latest policy can add one beyond top_k.
The builtin strategy has no random tie decisions; seed records provenance.

A replacement selector supplies matching name/version and the same select
interface. Its raw result must have the declared count, valid causal positions
and strict ascending order. Immutable score/position views prevent a plugin
from modifying the caller's score snapshot. The same manifest schema is used.

Query-head q maps to KV head q // (query_heads // kv_heads), after checking exact
divisibility. This is a logical model mapping, with no claim that a physical
probe can identify individual heads.

Config/sequence/score/manifest digests and expected_config bind identity,
layout, seed and fixed token history. Missing/unknown fields, duplicate config
JSON keys, altered digests, mismatched GQA, reordered positions and invalid
steps are refused. Digests provide integrity checks, not authentication.
Manifest validation without source scores cannot certify score ranking.
Tokenizer vocabulary validity and real model revisions remain model-adapter
responsibilities.

## Static cache and selected attention

The cache allocates K and V once, shape
[layers, KV heads, P+G-1, head_dim]. The complete cache byte count is checked
against 64 MiB before allocation. This version supports float32 and float64.
Writes validate one new [KV heads, head_dim] row and copy it into the next
position in place. Overwrites, gaps and unexpected shapes/dtypes/devices fail.
The implementation does not concatenate, clone or make the full cache
contiguous as history grows.

For each layer and step, the valid length must equal P+s. For each KV head,
the runner forms the union of its query heads' selected positions, gathers K
and V once from the source cache, then selects each query head's subset from
those small buffers. Dot products, scaling by sqrt(head_dim), softmax and value
aggregation operate on the selected rows. Only query, incoming row and selected
values are checked for finiteness; unselected cache contents are not scanned.
Nonfinite intermediate logits, probabilities or outputs fail.

A union may include the whole causal prefix for some strategies. The private
audit reports that actual logical union and theoretical selected-row bytes;
it does not promise a strict reduction for every valid manifest.

Tests check numerical agreement, unchanged output after unselected values
become NaN/Inf, stable pointers, source gather calls and absence of whole-cache
copy operations in the core. Those facts establish the implemented tensor
path. They do not measure GPU transaction counts, physical cache lines,
cache-set mappings, timing observability or probe accuracy.

## Oracle and probe separation

The probe request schema permits only opaque run/step IDs, an integer host
scheduling window and bounded own-buffer descriptors. Unknown fields at every
level are rejected, including positions, masks, labels, scores, model metadata,
paths and tensors. It returns a fresh primitive copy and imports no Torch.

Oracle selection, token history and evaluation results stay in separate
sidecars. The CPU smoke validates request examples but does not execute a
probe, allocate its buffers or emit simulated latency observations. A later
evaluator may join actual probe results to the oracle by opaque IDs after
collection. Identifier syntax and a strict interface are not an OS sandbox
or proof of buffer ownership.

## Remaining integration

No real LLaMA weights, tokenizer, RoPE/model forward, FP16/BF16 attention path,
generation adapter or physical probe is implemented here. The caller will
need authorized weights, byte-verified revisions and an audited model/cache
adapter. Actual hardware access and pattern observability need independent
measurement. The previous owned-GPU benchmark remains a failed post-idle
gate result with actual cross-context overlap unknown; this task does not
rerun it or change MPS, MIG, drivers or permissions.
