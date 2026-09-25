# Setup Notes

## Software Install

 1057  ls
 1058  uv
 1059  brew install uv
 1060  uv init
 1061  uv venv
 1062  ls
 1063  ls -lta
 1064  source .venv/bin/activate
 1065  uv add torth datasets tokenizers transformers safetensors
 1066  uv add torch datasets tokenizers transformers safetensors
 1067  history
 1068  uv add numpy matplotlib pandas tensorboard
 1069  ls
 1070  history
 1071  ls

## Software Usage

These have distinct purposes:

Library	Purpose
PyTorch- Model, training and automatic differentiation
Datasets - Download and stream Hugging Face datasets
Tokenizers - Train a modern BPE tokenizer
Transformers - Comparison models and generation utilities
Safetensors - Safe checkpoint storage
TensorBoard - Training graphs
Matplotlib/Pandas - Introspection reports

## Download Run

Basic download run:

uv run python -m vanus_toy_model_py.download_data