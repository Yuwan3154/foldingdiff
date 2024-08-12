#!/bin/bash

sudo apt-get update

# Upgrade existing packages
sudo apt-get upgrade -y

# Install required packages, try libgnutls28-dev if libgnutls-dev is not found
sudo apt-get install -y autoreconf autopoint libgnutls28-dev nettle-dev libgmp-dev libssh2-1-dev libc-ares-dev libxml2-dev zlib1g-dev libsqlite3-dev pkg-config libexpat1-dev libcppunit-dev autoconf automake autotools-dev libtool

# Navigate to the aria2 directory
cd path/to/aria2  # Replace with the actual path to the aria2 directory

# Run autoreconf to generate the configuration script
autoreconf -i

# Configure the build
./configure

# Build aria2
make

# Install aria2
sudo make install

echo "aria2 installation completed successfully."
