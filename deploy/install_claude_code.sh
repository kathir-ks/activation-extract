curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
source ~/.bashrc    # or restart terminal

nvm install --lts
nvm use --lts

node -v
npm -v

npm install -g @anthropic-ai/claude-code
