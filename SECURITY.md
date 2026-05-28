# Security Policy

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue for a
vulnerability.

- Use GitHub's **[Report a vulnerability](https://github.com/mindees/crypto-ai/security/advisories/new)**
  (Security → Advisories) to open a private report, **or**
- Open a regular issue **only** for non-sensitive, low-risk reports.

We aim to acknowledge reports within a few days.

## Scope & secrets

- This project ships **no secrets**. API keys (Kaggle, FRED, Etherscan, Telegram,
  Discord, SMTP) are read from environment variables / `.env` (gitignored) only.
- If you ever find a committed credential, report it privately and we will rotate it.
- Alert adapters are **disabled by default** and require both a config flag and
  environment credentials before they can send anything.

## A note on financial risk

This is **decision-support software, not investment advice**. "Security" here means
software security — it does **not** imply the model is safe to trade. See the
[disclaimer](README.md#license--disclaimer).
