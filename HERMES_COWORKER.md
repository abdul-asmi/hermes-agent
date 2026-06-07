# Hermes Coworker

This fork contains the production-oriented changes used by Abdul's Hermes
installation.

## Included changes

- Persistent, model-specific quota cooldowns
- Gemini daily request quotas skipped until midnight Pacific
- Short RPM/TPM limits retried after the provider's advertised delay
- Automatic fallback selection before the first API call of a new turn
- Earlier context compression for lower token use
- Gemini Embedding 2 plus Supabase pgvector memory
- Google Compute Engine deployment with Docker supervision and backups

## Model order

1. `gemini-3.5-flash`
2. `gemini-3.1-flash-lite`
3. `deepseek/deepseek-v3.2` through OpenRouter
4. `gemma-4-31b-it`
5. `gemma-4-26b-a4b-it`

OpenRouter requires `OPENROUTER_API_KEY`. If the key is absent, Hermes skips
that provider and continues down the chain.

## Why persistent cooldowns are needed

Hermes normally retries the preferred primary model at the beginning of every
new user turn. That is useful for a brief outage, but wasteful after a daily
quota is exhausted. Gateway agents can also be recreated between messages, so
an in-memory cooldown does not survive.

Cooldown state now lives at:

```text
~/.hermes/cache/model_cooldowns.json
```

The file contains model names, expiry times, and non-secret reasons. It does
not contain API keys or request content.

## Supabase memory

The `gemini_pgvector` plugin is under
`plugins/memory/gemini_pgvector/`. It embeds memories with
`gemini-embedding-2` at 768 dimensions and searches a private
`hermes.memory` pgvector table.

Embedding and database failures fail open: chat continues without recalled
memory. Completed memory saves run in the background.

Required environment variables:

```text
GOOGLE_API_KEY
DATABASE_URL
```

## Google Cloud

See [`deploy/google-cloud/README.md`](deploy/google-cloud/README.md). The core
agent runs on Compute Engine rather than Cloud Run because the WhatsApp bridge,
cron scheduler, and gateway require long-lived processes and persistent state.

Never commit `.env`, `auth.json`, WhatsApp session files, database passwords,
backups, or conversation exports.
