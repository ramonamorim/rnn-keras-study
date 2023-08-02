# rnn-keras-study
This project is just a study to use a RNN to classify text


## Structure of the data for this project
The structure of the data to be processed can follow this pattern of json: 

```
[
    {
        "lineText": "here the text you want to train your model",
        "key": "the_key_to_classify"
    },
    {
        "lineText": "here the text you want to train your model",
        "key": "the_key_to_classify"
    },
    ...
]
```

So you can create a huge array in one file with the strings with their respective key to train and test the model.

## RNN
This is a neural network for text sequence classification using an Embedding layer followed by an LSTM layer and two dense layers. The effectiveness of the network may depend on the specific details of the dataset and the task at hand.

```
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_sequence_length))
model.add(LSTM(100))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```



### Sequencial
model = Sequential(): This creates a Sequential model in Keras, which is a linear stack of layers. Each layer is added sequentially using the add() method.

### Embedding
model.add(Embedding(vocab_size, 100, input_length=max_sequence_length)): The first added layer is the Embedding layer. It maps integer indices (representing words) to dense vectors of size 100. vocab_size is the vocabulary size, representing the total number of unique words in the dataset. 100 is the dimension of the embedding vector, the size of the numerical vector each word will be mapped to. input_length is the maximum length of the input sequence, which defines how many tokens will be considered in each input example.

### LSTM
model.add(LSTM(100)): The second added layer is an LSTM (Long Short-Term Memory) layer. LSTMs are a type of recurrent neural network (RNN) designed to handle sequential data, such as sentences or documents. In this case, we have an LSTM layer with 100 units. These units are responsible for learning and capturing sequential patterns in the data.

### DENSE
model.add(Dense(128, activation='relu')): The third layer is a dense (fully connected) layer with 128 neurons. The activation function used is ReLU (Rectified Linear Activation), which introduces non-linearities into the network.

### DENSE
model.add(Dense(num_classes, activation='softmax')): The last layer is another dense layer with num_classes neurons, where num_classes is the number of classes or categories the model is trying to predict. In this case, the activation used is the softmax function, which converts the outputs into probabilities, indicating the probability of each class being the target class.

### COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']): Here, the model is compiled. The loss parameter is set to 'categorical_crossentropy', indicating that we are performing a multi-class classification task. The optimizer is set to 'adam', which is a popular optimizer used to adjust the network's weights during training. Finally, the accuracy metric is used to monitor the model's performance during training.
