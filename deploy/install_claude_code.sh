#!/bin/bash
#
# Install Claude Code CLI
#

set -e

echo "Installing NVM..."
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash

# Load nvm in current shell
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

echo "Installing Node.js LTS..."
nvm install --lts
nvm use --lts

echo "Node version: $(node -v)"
echo "NPM version: $(npm -v)"

echo "Installing Claude Code..."
npm install -g @anthropic-ai/claude-code

echo ""
echo "Installation complete!"
echo "Run 'claude' to start Claude Code"
