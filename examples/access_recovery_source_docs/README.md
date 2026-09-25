# Access Recovery Source Documentation

This directory contains the unannotated English-language source corpus for the
access-recovery example. The files are written as ordinary internal company
documentation rather than as a serialized process graph.

The corpus intentionally does not contain:

- graph node or edge identifiers;
- a complete transition table;
- aggregation candidates;
- source-fragment labels;
- expected hierarchy levels.

Process steps, conditions and dependencies are expressed through headings,
procedural language, responsibilities, exceptions and references between
documents. These are the signals that the later extraction stages must use to
identify activity cards and infer a process graph.

| File | Document type | Primary audience |
|---|---|---|
| D1 | Support policy | Service owners and support teams |
| D2 | Service Desk operating guide | Service Desk analysts |
| D3 | Identity recovery runbook | Identity analysts |
| D4 | Platform recovery runbook | Platform engineers |
| D5 | Verification, escalation and closure standard | Service Desk and incident managers |

The explicit model in `../access_recovery_process.md` is retained only as
ground truth for evaluating a graph reconstructed from this corpus. It must not
be supplied to the automatic extraction pipeline as source documentation.
