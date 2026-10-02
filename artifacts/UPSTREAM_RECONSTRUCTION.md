# Controlled upstream reconstruction milestone

This is an executable CPU software loop around generated toy workloads. It
creates reliable metadata, processes explicit probe observations and reference
boundaries, reconstructs representative token profiles, and feeds the existing
offline replay, QAI and ATR paths. It does not collect a physical side channel
or reproduce the Janus paper's reported attack accuracy.

## History and boundary

The publication base is 291d4322c154c8e9f47654b9f65ee5e692121546 on the existing
draft PR branch. Its complete tree was verified equal to the AutoDL
c7b2a60ad2e0793ca23a31d874bcc2aace85f861 tree. Work continues on local branch
codex/janus-upstream-controlled-v1, preserving the original local branch and
nine-commit audit history. No reset, force push or main merge was used.

## Evidence and numerical choices

References are Janus_EuroSys_27(1).pdf and the verified
Janus_EuroSys_27_Supplement(1).pdf. Main-paper pages below are PDF pages.

| Stage | Paper basis | Implemented meaning |
| --- | --- | --- |
| Kernel-time utilization proxy | Main p6, section 4.1 | cumulative kernel duration divided by window duration, in the same unit; not a hardware utilization counter or a probe count |
| Reference phase alignment | Supplement p3–4: collection-start alignment and nearest probe round to reference boundaries | Explicit clock anchors, nearest-round alignment, configured tie rule and maximum error; source remains oracle_aligned_phase |
| Prefill page proxy | Main p6–7, section 4.2.1 | Sum recorded eviction footprints over the prefill segment; relative_frequency_proxy is not an exact victim access count |
| Decoding page event | Main p7, section 4.2.2 | Translation reload latency strictly greater than the supplied calibrated threshold is 1; equality is 0 |
| Noise filtering | Main p7: neighboring windows and majority voting | All rounds within one oracle-aligned decoding step vote together; this grouping, minimum votes and tie rule are reconstruction choices; no cross-step sliding windows |
| Prefill token profile | Main p7–8, section 5.1: higher-frequency pages imply more concentrated participation | Configured power-law weights in declared token order, conserving each page's proxy mass; distribution and exponent rule are reconstruction choices |
| Decoding token profile | Main p8, section 5.2: local page density informs representative participation | Configured clipped neighborhood within one layer/head, ceil active-token count and prefix placement; these numerical choices are not specified by the paper |

The kernel ratio may exceed one when cumulative durations include concurrency.
The implementation does not clamp it or substitute interval-union occupancy.
It accepts an already aggregated kernel duration; aggregation/window alignment
from a physical kernel timeline is outside this module.

Trace-only sharp L2 drops and isolated TLB peaks are described qualitatively.
No autonomous detector or guessed detection constants are provided here.
Reference boundaries and gold labels are oracle annotations, separated from
probe observations throughout the contract.

## Source and input contracts

A janus.probe.run.v1 record declares stable run/sample/case/response IDs, a fixed
group split and assignment ID, an allocation epoch, clock anchors, logical
page/probe mapping, calibration, raw probe rounds, reference annotations and
gold metadata. The canonical JSON hash preserves types and ignores object key
order. Files and structural budgets are checked before copying.

The source classes have separate meanings:

| Source | Actual meaning |
| --- | --- |
| synthetic_probe_simulation | Numeric timing/eviction observations produced by a named simulator from a controlled CPU toy access schedule; never measured GPU latency |
| synthetic_layout / synthetic_calibration | Explicit toy page ranges, probe mapping and named simulator thresholds |
| oracle_annotation | Reference boundaries, tokenizer and gold labels from the controlled workload; never an observable probe channel |
| reconstructed_probe_profile / offline_probe_reconstruction | Derived representative token profiles / their provenance class; input source and oracle phase evidence stay explicit |
| synthetic_fixture / replay_derived | Generated bridge traces / their canonical replay derivatives; input source is recorded in provenance |
| physical_probe_recording | Future externally recorded probe data with collection identity and collector revision; no collector is executed here |
| calibrated_logical_mapping / physical_contention_calibration | Required declarations for processing physical records; not recovered physical addresses and not authentication of the declarations |

The mapping binds each logical monitored page to one attacker-owned probe,
layer/head order and a declared token range. Ranges must cover the token axis
without gaps or overlaps. Calibration identity and allocation epoch must match
the run and mapping. Calibration cannot silently carry across a restart or
allocation change. No victim KV contents or exact victim physical addresses
are synthesized; the paper does not require their recovery.

Raw round entries contain round/probe IDs, timestamps, translation reload latency
and recorded evicted-probe-line footprint only. Missing observations, gold/phase
fields disguised as raw, non-finite values, wrong units, duplicate IDs, ambiguous
ranges and incompatible source classes are rejected. Physical declarations
without mapping, calibration or epoch evidence are rejected.

## Reproducible controlled data

workload.py runs a deterministic toy KV access schedule and integer aggregation
with an explicit symbolic tokenizer. Its ordered table and revision are part of
the dataset. This CPU workload is not an LLM or a GPU collector.

Stable case/sample/response/step identities use the toy namespace, fixed case
ordinal and step index. They remain unchanged across seed/noise perturbations;
the generation seed and simulator revision are recorded separately. Case-group
hash assignment is fixed before access scheduling,
gold generation or probe simulation. The bundle stores the assignment contract
and digest; tampering with a run split without consistent evidence is rejected.
Hash buckets use the declared 80/10/10 policy, not a forced small-sample quota.

A conservative serialized-byte budget is checked before generating raw
observations. Bundle validation independently recomputes simulator observations
from configuration and fixed case/split inputs, binding declared noise and timing
parameters to the raw values. It does not recompute or overwrite oracle gold:
legitimate gold corrections remain possible, and tests verify they do not change
numeric features. This consistency check is not malicious-party authentication.


Gold token IDs come from the toy operation. The declared test-only zero-sum
policy supplies an OOV token for denominator checks; it does not enter the
training candidate vocabulary. Tokenizer revision, same-index step alignment
and excluded BOS/EOS rules are explicit. Noise affects simulated observations,
not the identities, split or gold calculation.

Named synthetic hit/miss latencies, thresholds and footprint budgets are
simulator constants. They are not physical calibration values inferred from
the paper. Both workload and numerical reconstruction configuration files
require their full explicit field set.

## Processing and failures

probe_reconstruction.py first aligns reference boundaries into the probe clock.
The first boundary must map to the first observed round. Segment starts must
remain distinct and increasing; the declared end must cover the final round.
Excessive skew, a collapsed boundary or insufficient decoding votes is an error.

Time arithmetic preserves the exact integer or already-parsed IEEE binary
value of each supported input. Raw-record integers retain the probe contract's
signed 64-bit budget; finite fractional floats are supported as their current
binary values, without recovering decimal precision lost during JSON parsing.
Clock subtraction, boundary/end addition, nearest-round distances, ties and
maximum-error comparisons use exact rational arithmetic throughout. Moving
integer timestamps and anchors to an epoch-scale clock does not discard small
relative errors or fractional ties.

Existing alignment-error and clock-offset JSON fields remain numeric. A
derived integer is emitted as an integer when it fits the signed 64-bit budget;
other derived values must round-trip exactly through a finite float. Values
that cannot be represented exactly in these fields are rejected with
ValidationError rather than rounded. This includes an unrepresentable
epoch-scale fractional clock offset. For example, an integer epoch anchor
plus a small relative 0.5 boundary supports an exact half-nanosecond tie;
an already-rounded absolute float timestamp cannot recover its lost fraction.
The aligned collection end must remain strictly later than the last round.

Source classes, calibration IDs and allocation epochs are input declarations
whose consistency is checked. They do not authenticate physical capture.


For prefill page proxy c_i in one layer/head, the configured exponent is
alpha_i = max_exponent * c_i / max(c). Token rank r receives normalized
(r+1)^(-alpha_i) weight times c_i. A zero page has zero mass; an all-zero head is
zero. This is a representative frequency-consistent profile, not recovered
attention probabilities or original top-k ranks.

For decoding, thresholded votes from every round in one oracle-aligned step
determine b_i. This grouping, minimum vote count and tie handling are explicit
reconstruction choices; no cross-step neighboring-window filter is implemented. Neighborhood density is the
mean of nearby b_j within the same ordered layer/head pages. An active page gets
max(1, ceil(density * token_count)) representative prefix tokens, capped at that
page's range; an inactive page gets zero. No gold labels select token positions.

All public operators reject invalid/non-finite parameters. Numerical values,
input meaning, configured policy and source provenance travel with outputs.

## Bridge and replay

upstream_bridge.py accepts the verified synthetic workload bundle. It produces
trace/replay records and preserves their case/response/step mapping. The QAI
manifest contains sample/case/split, attribute presence/label and NPZ path.
The ATR manifest and shared mapping_provenance.json retain tokenizer, explicit
response/step IDs and the complete gold step mapping. Physical records are rejected by this synthetic bridge before
output creation; the pure CPU record-processing interface remains separately
available for fully declared offline physical inputs.

The QAI adapter explicitly turns representative prefill profiles into a bounded
rank surrogate for the existing safe NPZ interface. Each run is divided by its own
global maximum, quantized with nearest-even rounding to uint16 [0, 65535],
and given a singleton query axis. The existing QAI histogram selects two
largest-value memberships and applies per-layer/head normalization.
Quantization/top-k selection is an adapter choice with information loss,
not a claim to recover legacy attention ranks. It only writes and reads its own generated numeric NPZ files.
Both numeric rank-payload equality across splits and the existing NPZ
byte-digest guard are retained; IDs and filenames never exempt duplicate
features. QAI train-label and ATR train-candidate contracts are frozen before
output creation, and existing output directories are preserved.

ATR consumes reconstructed layer/head/token profiles with explicit step IDs.
Train-only candidate construction remains unchanged. Missing predictions and
OOV tokens stay in the complete gold denominator. Both models use small CPU
training and this pipeline's safe state-dict checkpoints; no legacy model loads.

## Commands

Run from the repository with its existing environment:

~~~bash
python -m artifacts.janus_artifact.cli upstream-generate \
  --config artifacts/fixtures/controlled_workload.json \
  --output artifacts/test-output/controlled-workload.json

python -m artifacts.janus_artifact.cli upstream-reconstruct \
  --input artifacts/test-output/controlled-workload.json \
  --config artifacts/fixtures/probe_reconstruction.json \
  --output artifacts/test-output/controlled-profiles.json

python -m artifacts.janus_artifact.cli upstream-synthetic-smoke \
  --work-dir artifacts/test-output/upstream-example --device cpu
~~~

UPSTREAM_VALIDATION.md records actual regression and smoke results. Remaining
author evidence includes calibrated physical probe collection, trace-only
segmentation parameters, reconstruction parameters/validation and complete
author-matched experiment records. This milestone establishes software behavior,
not an end-to-end physical side-channel reproduction.
