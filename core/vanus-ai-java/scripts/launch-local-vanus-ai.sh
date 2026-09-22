#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module_dir="$(cd "${script_dir}/.." && pwd)"
checkpoint="${module_dir}/checkpoints/for-scatty-1m-2k.vanus"
dataset="${module_dir}/data/scatty-language/scatty.tsv"
server_args=()

while (($#)); do
  case "$1" in
    --checkpoint)
      [[ $# -ge 2 ]] || { echo "Missing value for --checkpoint" >&2; exit 2; }
      checkpoint="$2"; shift 2 ;;
    --data)
      [[ $# -ge 2 ]] || { echo "Missing value for --data" >&2; exit 2; }
      dataset="$2"; shift 2 ;;
    *) server_args+=("$1"); shift ;;
  esac
done

if ((${#server_args[@]} == 0)); then
  server_args=(-p 8086 -d "${module_dir}/data/server" --rate-limit 200 -h 127.0.0.1)
fi

quote_sbt() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

command="run server $(quote_sbt "$checkpoint") $(quote_sbt "$dataset")"
for argument in "${server_args[@]}"; do
  command+=" $(quote_sbt "$argument")"
done

cd "$module_dir"
exec sbt "$command"
