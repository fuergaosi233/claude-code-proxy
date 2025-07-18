list:
    @just --list

start_conf _file:
    #!/usr/bin/env bash
    uv run  claude-code-proxy  --env {{_file}}


updatedeps:
    uv export --no-editable --no-dev --no-hashes  --no-emit-project >requirements.txt
