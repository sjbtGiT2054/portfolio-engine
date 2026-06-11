# Investment Policy Statement — Portfolio Engine

**Owner:** Samuel Thompson
**Adopted:** June 2026 | **Review:** monthly alongside the macro assessment

## Objective

Maximise risk adjusted return (Sharpe ratio) on a long only equity ETF portfolio, expressing my top down macro views through constrained thematic baskets, as a learning vehicle for portfolio management and a credibility piece for buy side conversations.

## Mandate

- Instruments: pure play equity ETFs only (no bonds, money market, mutual funds or derivatives in phase 1)
- Direction: long only, fully invested, no leverage
- Universe: the approved list in `config.yaml` (currently 18 ETFs); additions require a written one line rationale here
- Rebalancing: monthly, driven by a full engine run
- Macro assessment: monthly, documented before each rebalance

## Risk appetite

- Single ETF maximum: 15%
- Basket exposures: within the min/max ranges in `config.yaml`
- Tolerance: equity market drawdowns are accepted; the correlation monitor and stress tests inform sizing, not market timing
- Hard rule: any FAIL on the tear sheet limit checks must be resolved before acting on weights

## Governance

- Decisions logged in `docs/Rebalance_Log.md` with rationale
- Views documented per theme in `docs/Macro_Assessment_Template.md` copies
- Limitations register in the README is read as standing context for every output

## Out of scope (phase gated)

Black Litterman views (phase 2), individual equities (phase 3), long short and any product/dashboard layer (phase 4, separate project). This tool is not investment advice and is used on my own account only.
