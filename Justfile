list:
    @just --list

start_conf _file:
    #!/usr/bin/env bash
    uv run  claude-code-proxy  --env {{_file}}


start:
    #!/usr/bin/env bash
    if [[ $(hostname) == *mini* ]] ; then
        just start_conf "./openroute.conf"
        
    else 
        just start_conf ".env_idealab"
    fi

whale:(start_conf ".env_whale")

idealab:(start_conf ".env_idealab")

openrouter:(start_conf ".env_openrouter")

updatedeps:
    uv export --no-editable --no-dev --no-hashes  --no-emit-project >requirements.txt
