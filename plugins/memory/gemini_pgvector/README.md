# Gemini pgvector memory

This Hermes memory provider uses:

- `gemini-embedding-2` with 768 dimensions
- PostgreSQL with the pgvector extension
- automatic top-5 semantic recall before each turn
- automatic storage of completed user and assistant turns

Required environment variables in `~/.hermes/.env`:

```env
GOOGLE_API_KEY=your_google_ai_studio_key
DATABASE_URL=your_supabase_postgres_connection_string
```

Use a restricted Supabase database role with only `SELECT` and `INSERT`
permissions on the private `hermes.memory` table. Do not use the Supabase
service-role key as the database password.

Activate it in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: gemini_pgvector
  gemini_pgvector:
    embedding_model: gemini-embedding-2
    embedding_dimension: 768
    top_k: 5
    auto_save: true
```
