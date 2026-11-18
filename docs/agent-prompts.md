# Agent prompt templates

Copy one block and fill in the bracketed fields. Give any performance baseline its workload, hardware, build options and QuACK revision.

## Understand the project

```text
Analysis only. Do not change code, documentation, configuration, Git state or the working tree.
Scope: [checkout, pipeline/profile, hardware, systems to include]
Desired behavior to explain: [what an operator should observe in one complete iteration]
Compatibility constraints: [hardware, ABI/wire versions, QuACK revision, legacy callers]
Acceptance criteria: [specific questions a reader should be able to answer]
Local validation: [read-only inspections and existing evidence allowed]
Remote validation: [hardware evidence available or commands to propose only]
Deliverables: [walkthrough, annotated file map, uncertainties and evidence]

Read AGENTS.md, README.md and docs/architecture.md. Trace one token from original
assignment to final output. Name the entry points, threads, streams, buffer
owners, readiness producers/consumers and shutdown joins. Identify which
settings come from code, JSON and the selected launcher. Cite source locations.
Label hardware claims by the evidence available.
```

## Understand one feature

```text
Analysis only. Do not edit code, tests, docs or configuration, or change Git state.
Scope: [feature, setting, profile, inputs and consumers]
Desired behavior to explain: [trigger, observable outcome and failure cases]
Compatibility constraints: [defaults, architecture, public contract, other features]
Acceptance criteria: [questions about supported combinations and ordering]
Local validation: [existing tests/results to inspect; read-only checks allowed]
Remote validation: [stages, hardware and evidence to identify or propose only]
Deliverables: [file-and-symbol trace, option matrix, tests, open questions]

Start from docs/features.md and docs/native-configuration.md. Follow the setting
from the launcher or caller through validation, native/Python boundaries and
the final consumer. Compare enabled, default, invalid and cancellation paths.
Identify dependencies on metadata coordinates, layout, publication and reuse.
Trace a real test, including the value or ordering assertion that would fail.
```

## Modify an existing feature

```text
Scope: [feature, entry points, files and callers affected]
Desired behavior: [current behavior, intended change and reason]
Compatibility constraints: [existing defaults, numerical tolerances, ABI, protocols,
architecture, concurrent policies, legacy configs and QuACK revision]
Acceptance criteria: [observable values, ordering, failures and performance target]
Local validation: [specific host/mock suites, configurations and expected results]
Remote validation: [target nodes, repeated numerical runs, profiling/baseline]
Deliverables: [implementation, relevant tests/docs and exact validation results]

Read AGENTS.md and trace both production and reference paths before editing.
Implement through the normal entry points. Preserve independent references.
Check partial/empty input, previous-generation state, timeout, abort and teardown
as applicable. Verify unaffected profiles as well as the requested change.
Report hardware validation separately and provide commands for unavailable nodes.
```

## Remove a feature

```text
Scope: [feature to retire, supported callers/releases, boundaries of removal]
Desired behavior: [operator-visible behavior for old configs/flags and replacement]
Compatibility constraints: [deprecation, mixed versions, old data, public ABI,
QuACK interfaces, reserved fields/protocol values and migration policy]
Acceptance criteria: [callers migrated, stale branches removed, retained behavior]
Local validation: [negative tests for retired settings, regression tests for survivors]
Remote validation: [supported topologies/architectures, numerical and timing checks]
Deliverables: [dependency analysis, implementation, compatibility notes, tests, docs]

Before deletion, trace dependencies in both directions through launchers, config
defaults/parsers, CMake, C ABI and ctypes, protocols, kernels, external consumers,
benchmarks and reference checks. Identify shared mechanisms still needed elsewhere.
State implications for old commands, persisted layouts and mixed deployments.
Implement the agreed compatibility behavior. Clean up orphan configuration,
code, tests, examples and documentation. Preserve adjacent coverage, update the
migration notes and re-scan the tree for active references to the removed feature.
```

## Implement a new feature

```text
Scope: [feature, entry points, supported profiles, boundaries]
Desired behavior: [enabled flow, default flow, invalid inputs and failure handling]
Compatibility constraints: [architecture, defaults, ABI/protocol, numerical rules,
QuACK revision, synchronization and lifetime contracts]
Acceptance criteria: [reference outputs, timing or ordering evidence, performance]
Local validation: [host invariants, parser/ABI tests, deterministic mock and error cases]
Remote validation: [Hopper/Blackwell, one/two nodes, repeated runs and baseline]
Deliverables: [working implementation, examples, tests, docs and validation record]

Read AGENTS.md. Locate the closest existing implementation and test. Explain
who owns each new buffer and which event, completion or prefix makes it safe to
consume or reuse. Implement the production entry point and validation together.
Cover supported interactions, error propagation and shutdown. Exercise a
non-default configuration and the existing default. Record any external change
required and the evidence for numerical correctness and performance.
```
