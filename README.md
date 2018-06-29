# keras-sentiment-analysis-api
A simple LSTM neural network model created based on [this tutorial](https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/).

To use the API, run webapp.py and enter http://127.0.0.1:5000/

Using a recurrent neural network for this task is a better approach than bag-of-words as the network can learn the context depending on the order of words. For instance:

* "I was very pesimistic and I thought it would be really bad. However, it turned out to be very, very good." gets a 53.70% positive score
* "I was very optimistic and I thought it would be really good. However, it turned out to be very, very bad." gets a 43.27% positive score

This test was performed after training the network for a few hours only, further training should improve the results.