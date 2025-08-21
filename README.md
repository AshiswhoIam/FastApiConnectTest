# FastApiConnectTest
FastApiConnectTest


# Deactivate current environment
conda deactivate

# Create a new environment with specific versions
conda create -n pokemon-api python=3.7.16 -c conda-forge
conda activate pokemon-api

# Install conda packages first
conda install -c conda-forge openssl

# Install requirements not using fast/uvi versions here.
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install tensorflow==2.4.0
pip install opencv-python==4.8.1.78
pip install requests==2.31.0
pip install python-multipart==0.0.6



# Old stuffs but I used these vs of fastapi and uvicorn
fastapi==0.68.0
uvicorn[standard]==0.15.0


tensorflow==2.4.0
numpy==1.19.5
pillow==8.4.0
python-multipart==0.0.5
opencv-python==4.5.3.56


# For rq i think

fastapi==0.68.0
uvicorn[standard]==0.15.0
tensorflow==2.4.0
opencv-python==4.8.1.78
requests==2.31.0
python-multipart==0.0.6
numpy==1.19.5





Using hugging space and docker, deployement works so far for fastapi.