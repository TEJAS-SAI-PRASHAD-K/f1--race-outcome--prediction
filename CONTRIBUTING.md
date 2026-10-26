# Contributing

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
git config core.hooksPath .githooks   # enables commit message validation
git config commit.template .gitmessage
```

## Commit conventions

This repo uses [Conventional Commits](https://www.conventionalcommits.org/),
enforced by `.githooks/commit-msg` — a POSIX shell script with no Node or
`commitlint` dependency. A malformed message is rejected at commit time.

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types** — `feat` `fix` `docs` `refactor` `perf` `test` `build` `ci` `chore`
`revert` `style`

**Scopes** — `db` `ingest` `features` `models` `backtest` `tune` `predict`
`config` `deps` `scripts` `docs` `repo`

**Rules**

- Header is at most 72 characters.
- Subject is lowercase, imperative mood (`add`, not `added`/`adds`), no
  trailing period.
- Scope is optional but preferred; it must come from the list above.
- `!` before the colon marks a breaking change (`feat(db)!: ...`), with a
  `BREAKING CHANGE:` footer explaining the migration.
- Body wraps at 72 characters and explains *why*. The diff already shows what.
- One logical change per commit.

**Examples**

```
feat(features): add rolling teammate qualifying delta
fix(ingest): skip sessions with no lap data instead of raising
docs(readme): document the recency half-life knob
```

## Branches

`feat/<scope>-<slug>` or `fix/<scope>-<slug>`. Work on a branch, not `main`.
