#!/bin/bash

# 設定根目錄路徑
ROOT_DIR="$(git rev-parse --show-toplevel)"

ENV_FILE="$ROOT_DIR/.env"
SECRETS_FILE="$ROOT_DIR/.secrets"

# 確保根目錄的 .env 檔存在
if [ ! -f "$ENV_FILE" ]; then
  echo ".env 檔不存在於根目錄：$ENV_FILE"
  exit 1
fi

# 將 .env 檔轉換為 act 可讀的 .secrets 檔
grep -v '^#' "$ENV_FILE" | grep '=' > "$SECRETS_FILE"

echo "已將 $ENV_FILE 轉換為 $SECRETS_FILE，供 act 使用。內容如下："
cat "$SECRETS_FILE"
