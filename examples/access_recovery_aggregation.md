# Hierarchical Aggregation Report

The base graph contains **18 nodes** and **21 edges**.

## L0 → L1

| Candidate | Atomic members | S_txt | S_ctx | S_flow | Q | Decision |
|---|---|---:|---:|---:|---:|---|
| C1 | v1, v2, v3, v4 | 0.097 | 1.000 | 0.857 | 0.762 | aggregated |
| C4 | v14, v15, v16, v17, v18 | 0.080 | 1.000 | 0.833 | 0.749 | aggregated |
| C3 | v10, v11, v12, v13 | 0.090 | 0.875 | 0.800 | 0.688 | aggregated |
| C2 | v6, v7, v8, v9 | 0.052 | 0.875 | 0.800 | 0.680 | aggregated |

Result: **5 nodes**, **5 edges**.

### Level mapping

| Source node | Target node |
|---|---|
| v1 | L1_A001 |
| v10 | L1_A003 |
| v11 | L1_A003 |
| v12 | L1_A003 |
| v13 | L1_A003 |
| v14 | L1_A002 |
| v15 | L1_A002 |
| v16 | L1_A002 |
| v17 | L1_A002 |
| v18 | L1_A002 |
| v2 | L1_A001 |
| v3 | L1_A001 |
| v4 | L1_A001 |
| v5 | L1_S001 |
| v6 | L1_A004 |
| v7 | L1_A004 |
| v8 | L1_A004 |
| v9 | L1_A004 |

## L1 → L2

| Candidate | Atomic members | S_txt | S_ctx | S_flow | Q | Decision |
|---|---|---:|---:|---:|---:|---|
| C5 | v1, v2, v3, v4, v5 | 0.086 | 1.000 | 0.800 | 0.737 | aggregated |
| C6 | v10, v11, v12, v13, v6, v7, v8, v9 | 0.088 | 0.438 | 0.800 | 0.513 | aggregated |
| Cx | v10, v5, v6, v7, v8, v9 | 0.058 | 0.583 | 0.800 | 0.565 | one or more alternatives are only partially covered in v5->v14 |

Result: **3 nodes**, **2 edges**.

### Level mapping

| Source node | Target node |
|---|---|
| L1_A001 | L2_A001 |
| L1_A002 | L2_S001 |
| L1_A003 | L2_A002 |
| L1_A004 | L2_A002 |
| L1_S001 | L2_A001 |
