sudo amazon-linux-extras enable epel
sudo yum install gcc openssl-devel bzip2-devel libffi-devel -y
sudo yum groupinstall "Development Tools" -y

# Download and compile Python 3.11
cd /usr/src
sudo curl -O https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz
sudo tar xzf Python-3.11.0.tgz
cd Python-3.11.0
sudo ./configure --enable-optimizations
sudo make altinstall
