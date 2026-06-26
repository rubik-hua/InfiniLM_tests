# ============================================================
# Terminal-Native AI Development Environment (Pure Ubuntu)
# Stack: Neovim + Avante.nvim + Claude CLI + Python/C++
# ============================================================

# Use the official pure Ubuntu image, goodbye Microsoft VS Code base
FROM ubuntu:22.04

# Suppress interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------
# 1. System Dependencies: Toolchains & CLI Utilities
# ----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic tools usually missing in pure Ubuntu
    curl wget git sudo ca-certificates locales zsh \
    # C/C++ toolchain, build systems, and debuggers
    build-essential cmake make ninja-build pkg-config \
    gcc g++ clang clangd clang-format \
    gdb valgrind \
    # Python runtime
    python3 python3-pip python3-venv \
    # Terminal utilities and search tools
    ripgrep fd-find unzip zip jq tree htop tmux \
    # Neovim (See below for AppImage alternative)
    # neovim \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------
# 1.5 Install Latest Neovim 0.11+ (avante.nvim requires >= 0.11)
# ----------------------------------------------------------
RUN curl -Lo /tmp/nvim.tar.gz https://github.com/neovim/neovim/releases/download/v0.11.0/nvim-linux-x86_64.tar.gz \
    && tar xzf /tmp/nvim.tar.gz -C /opt \
    && ln -sf /opt/nvim-linux-x86_64/bin/nvim /usr/local/bin/nvim \
    && rm /tmp/nvim.tar.gz

# ----------------------------------------------------------
# 2. Python Ecosystem: LSP, Formatters, and Debug Adapters
# ----------------------------------------------------------
# Upgrade pip first to avoid version-specific option errors
RUN pip3 install --no-cache-dir --upgrade pip

# Install Python packages (no --break-system-packages needed in 22.04, 
# but upgraded pip won't hurt)
RUN pip3 install --no-cache-dir \
    pyright black isort debugpy pynvim

# ----------------------------------------------------------
# 3. Node.js Runtime (Dependency for AI Agents)
# ----------------------------------------------------------
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------
# 4. AI Coding Agents
# ----------------------------------------------------------
RUN npm install -g @anthropic-ai/claude-code
RUN npm install -g @openai/codex 2>/dev/null || true

# ----------------------------------------------------------
# 5. Neovim Configuration with Avante.nvim
# ----------------------------------------------------------
COPY nvim-config/ /root/.config/nvim/

WORKDIR /root

# Install all lazy.nvim plugins (including avante.nvim)
RUN nvim --headless "+Lazy! sync" +qa 2>/dev/null || true

# Build avante.nvim's Rust backend
RUN nvim --headless "+AvanteBuild" +qa 2>/dev/null || true

# Install treesitter parsers for Python and C++
RUN nvim --headless "+TSInstallSync python cpp c lua" +qa 2>/dev/null || true

# ----------------------------------------------------------
# 7. Environment Defaults
# ----------------------------------------------------------
ENV EDITOR=nvim
ENV VISUAL=nvim

