#!/bin/bash

# Script to protect the Yang-Mills proof files from accidental modification
# Run this after verifying the proof is complete

echo "Protecting Yang-Mills proof files..."

# Make all Lean files in YangMillsProof read-only
find YangMillsProof -name "*.lean" -type f -exec chmod 444 {} \;

# Make Core foundation files read-only
find Core -name "*.lean" -type f -exec chmod 444 {} \;

# Keep build files writable
chmod 644 lakefile.lean YangMillsProof/lakefile.lean 2>/dev/null || true
chmod 644 lean-toolchain YangMillsProof/lean-toolchain 2>/dev/null || true

echo "✓ Proof files are now read-only"
echo "To make files writable again, run: find . -name '*.lean' -exec chmod 644 {} \;" 