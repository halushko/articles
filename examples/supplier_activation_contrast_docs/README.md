# Supplier activation contrast example

This is a compact executable corpus for comparing aggregation weights in the
web application. It describes one normal supplier-activation route with three
business stages owned by different roles and systems.

Use the same segmentation run and build the process model twice:

1. combined: `S_txt/S_ctx/S_flow = 0.2/0.4/0.4`;
2. topology only: `S_txt/S_ctx/S_flow = 0/0/1`.

The combined configuration should retain the three explicit stages. The
topology-only configuration may prefer a dense connected candidate crossing a
documented hand-over. The API response includes the active weights and
`derivation.config_sha256`; different weights must produce different process
model IDs and cache keys.

Verified deterministic result:

| Configuration | L0 | L1 | Accepted L1 hypothesis |
|---|---:|---:|---|
| Combined `0.2/0.4/0.4` | 12 | 3 | Three documented four-action stages |
| Topology only `0/0/1` | 12 | 4 | Two five-action cross-region groups plus two singletons |

The result is intentionally not judged by compression alone. The topology-only
run creates dense groups, but each accepted group crosses a documented business
handover. The combined run follows the reference partition by role, system,
operation wording, and flow.

The corpus is intentionally small so bounded cross-region candidate generation
can be inspected in full. No runtime rule checks these filenames, titles,
roles, systems, or operation wording.
