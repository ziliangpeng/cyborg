FROM node:22-slim

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# Install common dev tools, Python, and gh CLI
RUN apt-get update && apt-get install -y \
    git \
    curl \
    jq \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# Configure git to use gh CLI for credentials system-wide
# (user-level config is mounted from host, so we use /etc/gitconfig)
# Must run as root before switching to node user
RUN git config --system credential.helper '!gh auth git-credential'

# Shell aliases (in system bashrc so they survive home dir volume mount)
RUN echo 'alias csp="claude --dangerously-skip-permissions"' >> /etc/bash.bashrc \
    && echo 'alias x="exit"' >> /etc/bash.bashrc

# Use the existing node user (UID/GID 1000)
# This matches most macOS users' default UID
USER node
WORKDIR /home/node/code/cyborg

# Install Rust for the node user
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/node/.cargo/bin:${PATH}"

# Disable auto-update: updates are handled by rebuilding the container
ENV DISABLE_AUTOUPDATER=1

# Entry point runs claude, then drops into a shell
ENTRYPOINT ["bash", "-c", "claude --dangerously-skip-permissions \"$@\"; exec bash -li", "--"]
