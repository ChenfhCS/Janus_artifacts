# Owned GPU timing validation

This is a measured benchmark of two owned CUDA worker processes. It is not a
cache/TLB probe, model workload, semantic attack or paper reproduction.
No new GPU work was launched after the post-run aggregate gate failed.

## Frozen-source CPU regression

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s artifacts/tests -v
~~~

Actual final result: **288 tests passed in 54.602 seconds**, exit 0, no skips.
This retains all 257 prior tests and adds 27 CPU-only benchmark tests plus four
CLI tests. Benchmark tests mock device queries and worker processes; they do
not execute CUDA workers or a real nvidia-smi query.

Module SHA-256 tested before and after the final suite:
791b5591a8f08083afda29108e041f55f2a00b01a1bb9a8f6447958460e51824

Tests SHA-256:
76c65207cc6a67fcf5ae91072f64389546d2e1c5aea9bdff827b9859b31bb097

Private final log, relative to artifacts:
test-output/owned-gpu-timing-final-regression.log

Log SHA-256:
0467290edd328e552a92f21f01742e5728183d0d1e679b8e57f4b3bdee71dc01

The new CPU cases cover busy/unknown/ambiguous preflight with no workers,
strict identities, budgets and finite values, partial spawn, cleanup errors,
partial IPC frames, bounded query EOF/exit handling, worker bootstrap arguments,
the PyTorch UUID string conversion, missing/incorrect oracle results, allocator
peaks and incomplete cleanup, and a maximum eight-step result within 8192 bytes.
Independent source reviews checked timing semantics and process cleanup.

## Single persistent hardware run

~~~bash
python -m artifacts.janus_artifact.cli owned-gpu-timing-benchmark \
  --work-dir artifacts/test-output/owned-gpu-timing-final-v1 \
  --steps 3 --elements 65536 --timeout-seconds 20
~~~

**Final status: failed; CLI exit 2; overall cleanup state unknown.**
The three preflight snapshots each reported zero utilization and zero used
memory, Default compute mode and disabled MIG. Both worker device identities
matched the preflight identity. Exactly two owned CUDA workers were started.

| Measured item | Actual result |
| --- | --- |
| Worker processes / operations | 2 / 6 |
| Elements checked per operation | 65,536 |
| Total exact full-vector element comparisons | 393,216; all equal |
| Wall time through cleanup | 2.725202079862356 s |
| Wall time through report writing | 2.7258459627628326 s |
| Enclosing run budget | 30 s; met |
| Active declared CPU/GPU tensor buffers | 5,242,880 B (5 MiB) |
| Owned CUDA peak allocated bytes, sum | 2,097,152 B |
| Owned CUDA peak reserved bytes, sum | 4,194,304 B |
| CPU vectors plus peak CUDA reservation | 7,340,032 B (7 MiB); below 1 GiB |
| Each worker allocator after cleanup | allocated 0 B; reserved 0 B |
| Owned process exit confirmation | both confirmed dead; exit codes 0 and 0 |
| Post-run aggregate memory / utilization | 0 MiB / 13 percent |
| Host launch bracket intersections | observed; host activity only |
| Actual cross-context GPU overlap | unknown |
| Absolute CPU/GPU clock offset | unknown (null) |
| Effective UID / non-root deployment verified | 0 / false |

Own-process exits and allocator releases passed. The final sampled GPU
utilization was nonzero, so the strict device-idle cleanup gate failed.
Its cause is not attributed to another tenant, a leak or a particular workload.
No process enumeration, setting change, repeat GPU run or further device query
was used to resolve it. The overall failed/unknown result is preserved.

The six selected-operation event intervals, in milliseconds, were:
15.853568077087402, 0.18966400623321533, 0.04956800118088722,
16.759807586669922, 0.1802240014076233, 0.04867200180888176.
Their sum is 33.08150367438793 ms. These are context-local event intervals and
may include scheduling/host submission effects between event records; their
sum is not total GPU active time. Setup, copies and calibration are enclosed by
the wall-time budget but are not included in that selected-operation sum.

The host clock advertised 1e-9 s resolution; its empirical minimum positive
difference over 256 samples was 158 ns. The three empty-event observations
per worker were [0.020479999482631683, 0.0030720001086592674,
0.0030720001086592674] and [0.0030720001086592674,
0.0030720001086592674, 0.0030720001086592674] ms.
These observations do not determine physical clock precision or host/GPU
alignment. No simulated nanosecond tolerance was used. Integer oracle equality
is exact and has no tolerance.

The fresh runtime directory was created with mode 0700, report.json with 0600,
and private CLI/stderr files with 0600. No existing file or device permissions
were changed. No MPS, MIG or driver configuration was changed.

Private evidence, relative to artifacts:
- test-output/owned-gpu-timing-final-v1/report.json
- Disk snapshot SHA-256: 3f23bee80f27c57dd27bd931305b5c00054f38ee8ff659150f821b4217139386
- test-output/owned-gpu-timing-final-v1-cli.json
- CLI result SHA-256: 5a30e90c922a3408c09bff3f6dbbc35309297347787c689465eba4c02890f24e

The disk snapshot labels wall time through cleanup before persistence. The CLI
result additionally records wall time through report writing and confirms that
deadline too. Runtime JSON/logs, host timestamps, process identifiers and device
identity information are excluded from source archives.

## Source and publication boundary

The local benchmark branch was created from publication commit
bf9f8b490053d55c7a1de5da294cdb862ed06789 after a read-only fetch and complete
tree equality check against preserved source 2b5d361a56acfe5d6856794264003726d6443acc.
Both older local branches were retained. This task does not push to GitHub or
replace the Library source archive; the new source-only package is supplied for
parent review and publication.
