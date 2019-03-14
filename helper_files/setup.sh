# Requiremnets for this project

# basic libraries
sudo apt-get update
sudo apt-get install python3-pip -y
sudo -H pip3 install numpy 
sudo -H pip3 install scipy
sudo -H pip3 install pandas
sudo -H pip3 install --upgrade tensorflow

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

# resources for syllabification
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git

# sudo apt-get update
# sudo -H pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-linux_x86_64.whl  
# sudo -H pip3 install torchvision 
# sudo apt-get install gfortran -y
# sudo apt-get install libatlas-base-dev -y
# sudo apt-get install swig -y

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb \
    && sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub \
    && sudo apt-get update \
    && sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb \
    && sudo apt-get update \
    && sudo apt-get install -y cuda