# cd into data directory and execute shell script
mkdir -p cath_af
aria2c -x10 ftp://orengoftp.biochem.ucl.ac.uk/alphafold/cath-v4.3.0-model-organisms/cath-v4_3_0.alphafold-v2.2022-11-22.tsv -d cath_af
aria2c -x10 ftp://orengoftp.biochem.ucl.ac.uk/alphafold/cath-v4.3.0-model-organisms/cath-v4_3_0.alphafold-v2.2022-11-22.by_superfamily.tgz -d cath_af

# cd into the cath directory and untar the file
cd cath_af
tar -xzf cath-v4_3_0.alphafold-v2.2022-11-22.by_superfamily.tgz