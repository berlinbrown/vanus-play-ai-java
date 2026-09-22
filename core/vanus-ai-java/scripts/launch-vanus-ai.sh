#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module_dir="$(cd "${script_dir}/.." && pwd)"
deployment_jar="${module_dir}/target/scala-3.9.0/vanus-ai-java.jar"
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

if [[ ! -f "$deployment_jar" ]]; then
  echo "Deployment JAR not found: $deployment_jar" >&2
  echo "Build it first with: ${script_dir}/build-deployment.sh" >&2
  exit 1
fi

if ((${#server_args[@]} == 0)); then
  server_args=(-p 8086 -d "${module_dir}/data/server" --rate-limit 200 -h 127.0.0.1)
fi

cd "$module_dir"
exec "${JAVA_BIN:-java}" -Xmx2g --enable-native-access=ALL-UNNAMED \
  -jar "$deployment_jar" server "$checkpoint" "$dataset" "${server_args[@]}"
