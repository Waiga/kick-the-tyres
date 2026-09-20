# Architecture

Kick The Tyres has five focused stages:

1. `kick_the_tyres.cli` parses commands and controls filesystem and Git boundaries.
2. `kick_the_tyres.github_client` reads public GitHub metadata and text files;
   `kick_the_tyres.cache` stores reusable public responses locally.
3. `kick_the_tyres.scanner` examines local files without executing them.
4. `kick_the_tyres.scoring` converts observed evidence into usefulness and risk scores.
5. `kick_the_tyres.report` renders Markdown and escaped HTML reports.

`kick_the_tyres.models` holds the shared dataclasses and the `present` / `absent` /
`unknown` evidence type that every stage passes along unchanged.

Data flows in one direction from collection to evidence states, scoring, and reports.
An unavailable source remains `unknown`. Collection failures do not become negative
facts. A report records whether a static scan contributed to it, so a command that opens
no file reports risk as unknown instead of reporting no findings. Downloaded repositories
are never imported or executed by Kick The Tyres.
