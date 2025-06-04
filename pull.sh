#!/bin/bash

echo "🔄 正在拉取远程更新..."
git pull origin main

if [ $? -eq 0 ]; then
    echo "✅ 拉取成功！"
else
    echo "❌ 拉取失败，请检查冲突或网络连接。"
fi
