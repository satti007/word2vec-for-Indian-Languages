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

# sudo -H pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-linux_x86_64.whl  
# sudo -H pip3 install torchvision 
# sudo apt-get install gfortran -y
# sudo apt-get update
# sudo apt-get install libatlas-base-dev -y
# sudo apt-get install swig -y