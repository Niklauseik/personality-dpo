#!/bin/bash

echo "📦 添加所有更改..."
git add .

echo "📝 提交更改..."
git commit -m "Auto commit on $(date '+%Y-%m-%d %H:%M:%S')" || echo "✅ 无需提交"

echo "🚀 推送到 GitHub..."
git push -u origin main
