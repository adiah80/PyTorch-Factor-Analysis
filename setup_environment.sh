conda create --name $1 python=3.7 --yes
conda activate $1
echo "--- Environment created and activated."

### Install Standard Packages
echo "--- Installing Required Packages."
conda install --yes numpy pandas matplotlib tqdm 

### Install PyTorch
echo "--- Installing PyTorch."
conda install pytorch=1.4 cudatoolkit=10.1 -c pytorch