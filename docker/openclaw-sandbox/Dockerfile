FROM node:22-slim

# Install common dev tools, Python, and gh CLI (must come before npm install -
# openclaw needs git at install time)
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
    && curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl \
    && rm kubectl \
    && rm -rf /var/lib/apt/lists/*

# Install openclaw (after git is available)
ARG OPENCLAW_VERSION=latest
RUN npm install -g openclaw

# Configure git to use gh CLI for credentials system-wide
RUN git config --system credential.helper '!gh auth git-credential'

# Shell aliases (in system bashrc so they survive home dir volume mount)
RUN echo 'alias x="exit"' >> /etc/bash.bashrc

# Use the existing node user (UID/GID 1000)
USER node
WORKDIR /workspace

# Install Rust for the node user
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/node/.cargo/bin:${PATH}"

# Entry point starts the gateway daemon in foreground
ENTRYPOINT ["openclaw", "gateway", "--port", "18789", "--verbose"]
