#!/bin/bash

# Downloads the AlphaFold predictions for all of SwissProt as described
# Requires aria2c to be installed, see https://aria2.github.io

# Target directory containing AlphaFold Database files
target_directory=genie_afdb
mkdir -p "$target_directory"

# Check if the tar file has already been downloaded
tar_file=genie_afdb/swissprot_pdb_v4.tar
if [ ! -f "$tar_file" ]; then
    aria2c -x10 https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v4.tar -d genie_afdb
else
    echo "Tar file already exists. Skipping download."
fi

cd genie_afdb
# Unzip the file to a directory named genie_afdb
tar -xvf swissprot_pdb_v4.tar
