FROM node:22-slim

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# Install common dev tools and gh CLI
RUN apt-get update && apt-get install -y \
    git \
    curl \
    jq \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# Use the existing node user (UID/GID 1000)
# This matches most macOS users' default UID
USER node
WORKDIR /workspace

# Entry point runs claude with dangerous skip permissions
ENTRYPOINT ["claude", "--dangerously-skip-permissions"]
