# Data quality report

- Raw rows: **8418**
- Clean rows (silver): **8022**
- Quarantined rows: **396** (4.7%; gate at 15%)
- Vehicles: **24**
- Period: 2026-04-16 00:13:00 → 2026-10-12 23:16:00

## Why rows were quarantined

| Reason | Rows |
|---|---:|
| duplicate | 160 |
| soh_out_of_range | 87 |
| missing_soh | 82 |
| temp_out_of_range | 42 |
| future_timestamp | 25 |

## By manufacturer

| Source | Raw | Clean | Quarantined |
|---|---:|---:|---:|
| maker_b | 4297 | 4093 | 204 |
| maker_a | 4121 | 3929 | 192 |

## Injected defects (synthetic data only)

| Defect | Rows injected |
|---|---:|
| duplicates | 164 |
| soh_out_of_range | 82 |
| temp_out_of_range | 41 |
| missing_soh | 82 |
| future_timestamp | 24 |
