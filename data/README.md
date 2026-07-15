# Local data

Raw public and private data are intentionally excluded from Git. Committed
manifests record source, access date, license status, published checksum and the
locally verified SHA-256.

Expected paths:

- `raw/public/`: public development logs;
- `raw/benchmark/`: optional stress-test logs;
- `raw/private/kaleido_trace_port/`: pseudonymized Kaleido exports;
- `processed/`: regenerable canonical/prefix datasets;
- `manifests/`: committed provenance without source rows;
- `splits/`: frozen operation-level split manifests.

Public and synthetic results demonstrate pipeline competence only. They are not
evidence of Kaleido accuracy, savings, ROI or deployment success.
