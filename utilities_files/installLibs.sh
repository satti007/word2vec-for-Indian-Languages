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