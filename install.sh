sudo apt-get install libllvm-7-ocaml-dev libllvm7 llvm-7 llvm-7-dev llvm-7-doc llvm-7-examples llvm-7-runtime
sudo apt-get install build-essential libssl-dev libffi-dev python-dev lib32ncurses5-dev python-snappy
export LLVM_CONFIG=/usr/bin/llvm-config-7
git clone https://www.github.com/ildoonet/tf-pose-estimation
cd tf-pose-estimation
pip3 install cython
sudo python setup.py install
pip3 install -r requirements.txt
cd models/graph/cmu
bash download.sh