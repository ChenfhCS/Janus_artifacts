# Owned GPU timing and exact ground truth

This command measures short CUDA operations in two processes started by the
benchmark itself. It does not run a model or collect cache/TLB side-channel
observations. It is separate from the simulated upstream probe contract and
does not claim a Janus paper reproduction.

## Run

From the repository, use the existing PyTorch environment:

~~~bash
python -m artifacts.janus_artifact.cli owned-gpu-timing-benchmark \
  --work-dir artifacts/test-output/owned-gpu-timing-new \
  --steps 3 --elements 65536 --timeout-seconds 20
~~~

The output directory must be new. Existing files are preserved. The private
runtime report contains host timing, worker diagnostics and machine state;
test-output is ignored and excluded from source archives. A blocked preflight
or failed computation/cleanup returns nonzero and retains diagnostics.

Exactly two owned Python workers execute bounded integer operations, each with
independent CUDA allocations, CUDA events and a full CPU oracle comparison.
No model/checkpoint download or installation is required. Steps are at most 8;
elements are at most 1,048,576. The operation's int64 range is exact at those
limits. The comparison checks every element rather than a sampled checksum.

The declared tensor budget includes five int64 vectors per worker: GPU input
and output, CPU oracle input and output, and the copied result. Across both
workers this is 80 times elements bytes: 5 MiB by default, at most 80 MiB.
The workers also report their own CUDA allocator peaks. CPU vector bytes plus
the two peak reserved-memory counts must fit the 1 GiB buffer ceiling. CUDA
context/driver overhead is separate and is not measured as isolated device bytes;
idle pre/post aggregate snapshots cannot measure that overhead. The command refuses unsupported budgets and
uses a total wall-clock deadline of at most 30 seconds including preflight and
cleanup. The worker timeout can only tighten the bounded run.
The disk report labels wall time measured through cleanup before persistence;
the printed CLI result additionally measures through report writing. A write
that breaches the deadline is a failure, with diagnostic persistence attempted.

## Resource gate and lifecycle

Before starting a CUDA worker, the parent checks three aggregate device
snapshots. It requires one unambiguously selected device, unchanged identity,
Default compute mode, disabled MIG, zero sampled GPU utilization and zero
reported used memory. Busy, unsupported or uncertain state stops before worker
creation. These sampled aggregate checks do not establish an exclusive lease.

Device UUIDs are compared as canonical strings. PyTorch 2.8 exposes a UUID
object with a string conversion; see its [official binding](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/cuda/Module.cpp#L986).
Unknown or malformed identities are refused before worker creation.

The parent does not create a CUDA context. It starts only its two worker
processes, retains their handles immediately, and uses bounded IPC, waits and
cleanup. An error, timeout or partial spawn takes the same cleanup path:
terminate remaining owned workers, bounded wait, kill if necessary, bounded
wait again. Every owned handle must be confirmed exited. Successful workers
free their tensors and clear their own allocator cache before reporting
allocation/reservation counters. A final aggregate device query checks the
post-run state; uncertainty is reported rather than claimed as verified cleanup.

The command does not enumerate other tenants' processes, inspect their memory,
query administrative performance counters, or modify MPS, MIG, drivers,
permissions or device configuration. It does not enable MPS to manufacture
overlap. Preflight cannot eliminate races with another client beginning work
later. No attribution of another process is attempted.

## Timing contract

Each operation has explicit sample and step IDs. Its host records use integer
CPU monotonic timestamps around start-event recording, end-event recording and
synchronization. CUDA elapsed_ms uses events in that worker's own context.

| Evidence | Meaning |
| --- | --- |
| Host launch interval | CPU time enclosing CUDA event/kernel submission |
| Host completion bracket | CPU time enclosing submission through synchronization |
| CUDA event elapsed_ms | Relative elapsed time within that worker's context |
| Host interval intersections | Host activity intersections only |
| Actual cross-context GPU overlap | Always unknown in this benchmark |
| Absolute CPU/GPU clock offset | Unknown; reported as null |

Independent CUDA event durations do not provide a common absolute GPU timebase.
Event intervals may include scheduling effects and do not measure exclusive
device occupancy. Their sum covers the selected operations, not all setup,
allocation, copy or calibration work; the wall-clock deadline encloses the run.
Host launch/completion intersections cannot prove actual GPU kernel overlap.
A shared GPU timebase or appropriate profiler evidence would be needed for that
claim; neither is introduced by this command.

The report separates the host clock's advertised resolution from empirical
minimum positive monotonic_ns differences. Empty CUDA event pairs preserve
finite, nonnegative observed intervals including zero and their host brackets.
Those observations do not establish the physical GPU timer resolution or
CPU/GPU clock alignment. No simulated 5 ns tolerance is applied to hardware.

Ground truth is an exact full-vector CPU/GPU int64 equality test, with no
floating-point tolerance. Host/event timings must be finite and nonnegative,
and host timestamps must be ordered. Any mismatch fails the run.

The SSH effective UID is recorded faithfully. A run as root does not verify
deployment under the paper's ordinary-user threat model; this benchmark always
reports non_root_deployment_verified=false.
