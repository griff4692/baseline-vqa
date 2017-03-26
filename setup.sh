sudo nvidia-smi -pm 1
sudo nvidia-smi --auto-boost-default=0
sudo nvidia-smi -ac 2505,875
cd data/
mkdir glove

aws s3 cp s3://visualqa777/glove/glove.zip glove.zip
unzip glove.zip

