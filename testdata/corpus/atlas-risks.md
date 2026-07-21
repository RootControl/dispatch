# Project Atlas — Risk Register

## Schedule risk

The infrastructure migration slipped a quarter, compressing the window for
end-to-end testing. Mitigation is to start reconciliation testing against a
staging replica rather than waiting for the production cutover.

## Vendor risk

The statement mailer depends on a single external vendor whose contract renews
annually. Renewal was signed in January. A second vendor was evaluated but not
retained, so there is no live fallback.

## Data quality

Roughly two percent of historical invoices carry malformed line items inherited
from an earlier migration. These need cleanup before reconciliation can be
trusted, and the cleanup is not currently staffed.
