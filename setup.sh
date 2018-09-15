# Requiremnets for this project

# basic libraries
sudo apt-get update
sudo apt-get install python3-pip -y
sudo -H pip3 install numpy 
sudo -H pip3 install scipy
sudo -H pip3 install pandas

# crawling library for preprocessing
sudo apt-get update
sudo -H pip3 install bs4
sudo -H pip3 install requests

# polygot library for preprocessing
sudo apt-get update
sudo apt-get install python3 libicu-dev -y
sudo -H pip3 install pycld2
sudo -H pip3 install pyicu
sudo -H pip3 install polyglot
sudo -H pip3 install morfessor


curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh
sudo apt-get install nvidia-384 nvidia-modprobe -Y
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
chmod +x cuda_9.0.176_384.81_linux-run
./cuda_9.0.176_384.81_linux-run --extract=$HOME



# sudo -H pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-linux_x86_64.whl  
# sudo -H pip3 install torchvision 
# sudo apt-get install gfortran -y
# sudo apt-get update
# sudo apt-get install libatlas-base-dev -y
# sudo apt-get install swig -y


