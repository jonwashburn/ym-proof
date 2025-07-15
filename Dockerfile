# Yang-Mills Proof - Hermetic Lean 4 Build Environment
# This Dockerfile provides a consistent, reproducible build environment
# for the Yang-Mills proof with Lean 4.12 and Mathlib 4.12

FROM ubuntu:22.04

# Metadata
LABEL maintainer="Yang-Mills Proof Project"
LABEL description="Hermetic Lean 4 build environment for Yang-Mills proof"
LABEL version="1.0.0"

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Core build tools
    build-essential \
    curl \
    git \
    unzip \
    # Compression tools
    zstd \
    gzip \
    # Development utilities
    vim \
    less \
    bash-completion \
    # Network tools
    wget \
    ca-certificates \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -s /bin/bash lean \
    && usermod -aG sudo lean \
    && echo "lean ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to non-root user
USER lean
WORKDIR /home/lean

# Install elan (Lean version manager)
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
ENV PATH="/home/lean/.elan/bin:$PATH"

# Install specific Lean version
COPY --chown=lean:lean lean-toolchain .
RUN elan toolchain install $(cat lean-toolchain) \
    && elan default $(cat lean-toolchain)

# Set up working directory
WORKDIR /workspace

# Copy project files
COPY --chown=lean:lean . .

# Pre-download dependencies and build mathlib cache
RUN lake update

# Try to get community mathlib cache first (fastest)
RUN lake exe cache get || echo "Community cache unavailable, will build locally"

# Build mathlib if cache wasn't available
RUN lake build Mathlib.Data.Real.Basic Mathlib.Analysis.InnerProductSpace.Basic

# Create cache optimization script
RUN echo '#!/bin/bash' > /usr/local/bin/optimize-cache \
    && echo 'find /workspace/.lake -name "._*" -delete 2>/dev/null || true' >> /usr/local/bin/optimize-cache \
    && echo 'find /workspace/.lake -name "*.tmp" -delete 2>/dev/null || true' >> /usr/local/bin/optimize-cache \
    && echo 'echo "Cache optimized"' >> /usr/local/bin/optimize-cache \
    && sudo chmod +x /usr/local/bin/optimize-cache

# Set up environment variables
ENV LEAN_PATH="/workspace/.lake/build/lib"
ENV LAKE_NO_TIMEOUT=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD lake --version || exit 1

# Default command
CMD ["/bin/bash"]

# Build instructions:
# docker build -t yang-mills-lean .
# docker run -it -v $(pwd):/workspace yang-mills-lean
#
# To build the proof:
# docker run -it yang-mills-lean bash -c "cd /workspace && lake build"
#
# To run verification:
# docker run -it yang-mills-lean bash -c "cd /workspace && ./verify_roots_complete.sh" 