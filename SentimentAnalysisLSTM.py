import torch.nn as nn


class SentimentAnalysisLSTM(nn.Module):
    '''
    RNN Model for our Sentiment Analysis Task
    '''

    def __init__(self, vocabSize, outputSize, embeddingDim, hiddenDim, noOfLayers, dropProbability=0.5):
        '''
        Init model by creating Layers:
          - Embedding Layer: that converts our word tokens (integers) into embedding of specific size
          - LSTM Layer: defined by hidden state dims and number of layers
          - Fully Connected Layer: that maps output of LSTM layer to a desired output size
          - Sigmoid Activation Layer: that turns all output values in a value between 0 and 1
          - Output: Sigmoid output from the last timestep is considered as the final output of this network
        '''

        super().__init__()

        # defining layers
        self.outputSize = outputSize
        self.noOfLayers = noOfLayers
        self.hiddenDim = hiddenDim

        # embedding and LTSM layers using pyTorch nn for training and using Nueral Networks
        self.embedding = nn.Embedding(vocabSize, embeddingDim)
        self.lstm = nn.LSTM(embeddingDim, hiddenDim, noOfLayers, dropout=dropProbability, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hiddenDim, outputSize)
        self.sig = nn.Sigmoid()

    def passForward(self, xInput, hidden):
        """
        method makes a forward pass on our model on some xInput and hidden state
        """

        batchSize = xInput.size(0)

        # defining embedding and ltsmOutput
        embeds = self.embedding(xInput)
        lstmOutput, hidden = self.lstm(embeds, hidden)

        # stack ltsm outputs
        lstmOutput = lstmOutput.contigous().view(-1, self.hiddenDim)

        # dropout and fully connected layer
        out = self.dropout(lstmOutput)
        out = self.fc(out)

        # sigmoid function
        sigmoidOutput = self.sig(out)
        sigmoidOutput = sigmoidOutput.view(batchSize, -1)
        sigmoidOutput = sigmoidOutput[:, -1]

        return sigmoidOutput, hidden

    def initHidden(self, batchSize):
        """
        Initializes the hidden state
        """

        # create two tensors with size of noOfLayers * batchSize * hiddenDim,
        # which will be init to zero for hidden state and cell state of LSTM

        weight = next(self.parameters()).data

        hidden = (weight.new(self.noOfLayers, batchSize, self.hiddenDim).zero_(),
                  weight.new(self.noOfLayers, batchSize, self.hiddenDim).zero_())

        return hidden
