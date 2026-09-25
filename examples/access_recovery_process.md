# User Access Recovery Process

This document is a machine-readable representation of the control process used
to demonstrate hierarchical aggregation. The process graph is defined by the
two Markdown tables below. Narrative documentation may be kept in the same file
or in linked Markdown artifacts.

## Process Nodes

| id | operation | role | system | source_fragment_ids |
|---|---|---|---|---|
| v1 | Receive user request | Service Desk | ITSM | D1-F01 |
| v2 | Register incident | Service Desk | ITSM | D1-F02 |
| v3 | Determine incident priority | Service Desk | ITSM | D1-F03 |
| v4 | Classify incident | Service Desk | ITSM | D1-F04 |
| v5 | Determine failure type | Service Desk | ITSM | D1-F05 |
| v6 | Check account status | Identity Analyst | IAM | D2-F01 |
| v7 | Check authentication policy | Identity Analyst | IAM | D2-F02 |
| v8 | Check external identity provider availability | Identity Analyst | External IdP | D2-F03 |
| v9 | Restore user access | Identity Analyst | IAM | D2-F04 |
| v10 | Check service availability | Platform Engineer | Monitoring | D3-F01 |
| v11 | Check application status | Platform Engineer | Monitoring | D3-F02 |
| v12 | Check service dependencies | Platform Engineer | Monitoring | D3-F03 |
| v13 | Restore software platform | Platform Engineer | Orchestrator | D3-F04 |
| v14 | Verify recovery result | Service Desk | ITSM | D1-F06; D5-F01 |
| v15 | Determine incident outcome | Service Desk | ITSM | D1-F07; D5-F02 |
| v16 | Escalate unresolved incident | Service Desk | ITSM | D1-F08; D5-F03 |
| v17 | Notify user of result | Service Desk | ITSM | D1-F09; D5-F04 |
| v18 | Close incident | Service Desk | ITSM | D1-F10; D5-F05 |

## Process Edges

| id | source | target | type | condition |
|---|---|---|---|---|
| e1 | v1 | v2 | sequence | Request received |
| e2 | v2 | v3 | sequence | Incident registered |
| e3 | v3 | v4 | sequence | Priority assigned |
| e4 | v4 | v5 | sequence | Classification completed |
| e5 | v5 | v6 | conditional | Identity-related failure |
| e6 | v5 | v10 | conditional | Platform-related failure |
| e7 | v6 | v7 | sequence | Account exists |
| e8 | v7 | v8 | conditional | External federation check required |
| e9 | v8 | v9 | sequence | External IdP is available |
| e10 | v7 | v9 | conditional | Local policy error confirmed |
| e11 | v9 | v14 | sequence | Access restored |
| e12 | v10 | v11 | sequence | Service is available for diagnostics |
| e13 | v11 | v12 | conditional | Dependency check required |
| e14 | v12 | v13 | sequence | Faulty dependency identified |
| e15 | v11 | v13 | conditional | Local application fault confirmed |
| e16 | v13 | v14 | sequence | Platform restored |
| e17 | v14 | v15 | sequence | Verification completed |
| e18 | v15 | v17 | conditional | Recovery successful |
| e19 | v15 | v16 | conditional | Recovery unsuccessful |
| e20 | v16 | v17 | sequence | Escalation registered |
| e21 | v17 | v18 | sequence | User notified |

## Aggregation Candidates

The candidate set is stated explicitly because candidate generation is a
separate stage from candidate scoring and selection. Members always refer to
atomic L0 node identifiers, including candidates evaluated at higher levels.

| candidate_id | target_level | name | members | purpose |
|---|---|---|---|---|
| C1 | 1 | Intake and classification | v1; v2; v3; v4 | aggregation |
| C2 | 1 | Access recovery | v6; v7; v8; v9 | aggregation |
| C3 | 1 | Platform recovery | v10; v11; v12; v13 | aggregation |
| C4 | 1 | Verification and completion | v14; v15; v16; v17; v18 | aggregation |
| C5 | 2 | Incident qualification | v1; v2; v3; v4; v5 | aggregation |
| C6 | 2 | Diagnosis and recovery | v6; v7; v8; v9; v10; v11; v12; v13 | aggregation |
| Cx | 2 | Partially covered split-join region | v5; v6; v7; v8; v9; v10 | diagnostic |
