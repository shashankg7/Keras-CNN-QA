# Intro
This code is remplementation of  Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks. SIGIR, 2015 in Keras. 

This code is adapted from repo. https://github.com/aseveryn/deep-qa.

# Depdendencies
- python 2.7+
- numpy
- theano/tensorflow
- keras

# Embeddings 

The pre-initialized word2vec embeddings have to be downloaded from [here](https://drive.google.com/folderview?id=0B-yipfgecoSBfkZlY2FFWEpDR3M4Qkw5U055MWJrenE5MTBFVXlpRnd0QjZaMDQxejh1cWs&usp=sharing).


# Steps to run
To run the model, first run parsing file
>$ python parse.py

Then run 
>$ python ltr_cnn.py

# TO-DO
Currently support for external features (overlapping words from paper) is not supported. 

If anyone is interested, let me know, or you are most welcome to send a PR.
