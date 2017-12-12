nasnet_mobile_url=https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_mobile_04_10_2017.tar.gz

nasnet_large_url=https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz

wget ${nasnet_mobile_url}

wget ${nasnet_large_url}

mv nasnet-a_mobile_04_10_2017.tar.gz pre-trained/
mv nasnet-a_large_04_10_2017.tar.gz pre-trained/

mkdir pre-trained/nasnet-a_mobile_04_10_2017
mkdir pre-trained/nasnet-a_large_04_10_2017

tar -xzvf pre-trained/nasnet-a_mobile_04_10_2017.tar.gz -C pre-trained/nasnet-a_mobile_04_10_2017
tar -xzvf pre-trained/nasnet-a_large_04_10_2017.tar.gz -C pre-trained/nasnet-a_large_04_10_2017


