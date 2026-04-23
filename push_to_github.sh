#!/bin/bash
cd "$(dirname "$0")"

echo "Cleaning up old .git if it exists..."
rm -rf .git

echo "Initializing fresh repo..."
git init -b main
git config user.email "lukerborg15@gmail.com"
git config user.name "Luke"

echo "Staging files..."
git add -A

echo "Committing..."
git commit -m "Initial commit: strategy lab ML pipeline"

echo "Adding remote..."
git remote add origin https://github.com/lukerborg15-coder/Machine-Learning-Models.git

echo ""
echo "Ready to push. You will be prompted for:"
echo "  Username: lukerborg15-coder"
echo "  Password: your GitHub Personal Access Token (NOT your password)"
echo "  Generate one at: https://github.com/settings/tokens (check 'repo' scope)"
echo ""
git push -u origin main

echo ""
echo "Done! Check https://github.com/lukerborg15-coder/Machine-Learning-Models"
read -p "Press Enter to close..."
