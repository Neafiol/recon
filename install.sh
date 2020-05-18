sudo apt-get update
sudo apt-get install git libllvm-7-ocaml-dev libllvm7 llvm-7 llvm-7-dev llvm-7-doc llvm-7-examples llvm-7-runtime build-essential libssl-dev libffi-dev python-dev lib32ncurses5-dev python-snappy

curl -O https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
sha256sum Anaconda3-5.1.0-Linux-x86_64.sh
bash Anaconda3-5.1.0-Linux-x86_64.sh
conda install -c conda-forge tensorflow=1.14

apt update && apt install -y libsm6 libxext6


export LLVM_CONFIG=/usr/bin/llvm-config-7
git clone https://github.com/Neafiol/recon
pip3 install cython
cd recon
git clone https://www.github.com/ildoonet/tf-pose-estimation
pip3 --default-timeout=1000 install -r requerements.txt
cd tf-pose-estimation
python3 setup.py install
sudo python setup.py install
pip install -r requerements.txt
cd models/graph/cmu
bash download.sh