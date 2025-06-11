#!/bin/bash

# 기본 usage 함수
usage() {
  echo "Usage: $0 --path <directory>"
  exit 1
}

# 인자 파싱
while [[ $# -gt 0 ]]; do
  case "$1" in
    --path)
      DIR="$2"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

# 경로가 비어있으면 종료
if [[ -z "$DIR" ]]; then
  usage
fi

# 디렉토리 존재 확인
if [[ ! -d "$DIR" ]]; then
  echo "Error: Directory '$DIR' not found."
  exit 1
fi

# 실행
for file in "$DIR"/*.sh; do
  if [[ -f "$file" ]]; then
    echo "Running: $file"
    bash "$file"
    if [[ $? -ne 0 ]]; then
      echo "Error running $file"
    fi
  fi
done