# Security policy

FlowTwin is an offline/read-only MVP. It must never receive source-system write
credentials or control operational or safety systems.

## Data handling

- Keep Kaleido exports under `data/raw/private/`; that path is ignored by Git.
- Pseudonymize customer, person, vessel, project and resource identifiers before
  data leaves the agreed Kaleido environment.
- Treat notes, photographs and document references as potentially personal and
  confidential. They are excluded from the default feature set.
- Obtain written GDPR roles, purpose, retention and deletion terms before pilot
  ingestion.
- Never train across customers without explicit written agreement.

## Secrets and access

- Use minimum-privilege, read-only credentials from an approved secret store.
- Do not place credentials in configs, logs, artifacts, fixtures or the browser.
- The reference deployment binds to localhost by default and exposes no source
  mutation endpoint.

## Reporting

Report a suspected vulnerability privately to the EVOCON project owner. Include
the affected version, reproduction steps and impact. Do not attach confidential
customer data.
