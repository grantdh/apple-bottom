#!/bin/bash
# apple-bottom repo initialization script
# Run this once after cloning or creating the repo locally

set -e

echo "🍎 Initializing apple-bottom..."

git init
git add .
git commit -m "feat: initial BLASphemy — apple-bottom scaffold

Metal-native BLAS for Apple Silicon.
DGEMM Metal kernel stub + Swift interposition layer.

Copyright 2026 Technology Residue
Author: Grant David Heileman, Ph.D."

echo ""
echo "✅ Done. Now create the GitHub repo and run:"
echo ""
echo "  git remote add origin https://github.com/TechnologyResidue/apple-bottom.git"
echo "  git branch -M main"
echo "  git push -u origin main"
echo ""
echo "🤘 BLASphemy is go."
