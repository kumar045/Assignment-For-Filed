# 1. System Updates and Dependencies
sudo apt-get update
sudo apt-get install -y build-essential
sudo apt-get install -y libstdc++-12-dev
sudo apt-get install -y clang llvm
sudo apt-get install -y libc++-dev libc++abi-dev
sudo apt-get install -y llvm-14 llvm-14-dev
sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

# 2. Conda Environment Setup
conda create -n bitnet-cpp python=3.9 -y
conda activate bitnet-cpp

# 3. Clone Repository
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet

# 4. Python Dependencies
pip install -r requirements.txt
pip install cmake

# 5. Download Model
huggingface-cli download HF1BitLLM/Llama3-8B-1.58-100B-tokens --local-dir models/Llama3-8B-1.58-100B-tokens

# 6. Run Setup
python setup_env.py -md models/Llama3-8B-1.58-100B-tokens -q i2_s

# 7. Run Inference
python run_inference.py -m models/Llama3-8B-1.58-100B-tokens/ggml-model-i2_s.gguf -p "Daniel went back to the the the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?\nAnswer:" -n 6 -temp 0
FacebookTwitterEmailLinkedIn
