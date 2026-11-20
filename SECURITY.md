# Security notes

- Keep real API keys in a local `.env` file that is excluded from Git.
- Commit only `.env.example` with placeholder values.
- Do not commit `chroma db/`, uploaded PDFs, local databases, or private datasets.
- Uploaded files and vector stores may contain third-party or sensitive content.
- Test public demo deployments with non-sensitive sample documents only.
- If a key was ever committed or pushed, revoke/rotate it first, then follow GitHub's sensitive-data removal documentation. Adding `.gitignore` later does not erase Git history.
