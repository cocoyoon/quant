
from re import L
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.layers import (Input, Dense, LSTM, Conv2D, 
                                     BatchNormalization, Dropout, 
                                     MaxPooling2D, Flatten)
import tensorflow as tf
import numpy as np
import threading

from tensorflow.python.ops.gen_math_ops import Max


graph = tf.compat.v1.get_default_graph()
session = tf.compat.v1.Session()

class Neural_Network:

    """
        In A3C, we need to use multi-thread, so to prevent collapsing between thread, we need Lock class.
        graph: Space where Neural Network is defined.
        session: Space where Neural Netowrk is excuted.
    """

    lock = threading.Lock()

    def __init__(self,
    
        input_dim = 0,
        output_dim = 0,
        lr = 0.001,
        shared_network = None,
        activation = "sigmoid",
        loss = "mse",
    ):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.shared_network = shared_network
        self.activation = activation
        self.loss = loss
        self.model = None

    def predict(self, input):

        """
            Generates output predictions(Buy, Sell) for the input samples.
        """

        with self.lock:
            with graph.as_default():
                if session is not None:
                    set_session(session = session)

                return self.model.predict(input).flatten()

    def get_loss(self, input, label):

        """
            Return loss.
        """
        
        loss = 0.0
        with self.lock:
            with graph.as_default():
                if session is not None:
                    set_session(session = session)

                loss = self.model.train_on_batch(x=input, y=label)

        return loss

    def save_weights(self, path):

        if path is not None and self.model is not None:

            self.model.save_weights(filepath=path, overwrite=True)

    def load_weights(self, path):

        if path is not None and self.model is not None:

            self.model.load_weights(filepath=path)

    @classmethod
    def get_shared_network(cls, network="dnn", num_steps=1, input_dim=0):

        """
            Each of the NN(DNN,LSTM,CNN) has 'get_network' method
        """
        with graph.as_default():

            if session is not None:

                set_session(session=session)

            if network == "dnn":
                return cls.get_newtwork(Input(input_dim, ))

            elif network == "lstm":
                return cls.get_network(Input((num_steps, input_dim)))

            elif network == "cnn":
                return cls.get_network(Input((1, num_steps, input_dim)))

class DNN(Neural_Network):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        with graph.as_default():

            if session is not None:
                set_session(session = session)
                
            input = None
            output = None

            if self.shared_network is None:

                input = Input((self.input_dim, ))
                output = self.get_network(input = input).output

            else: 

                input = self.shared_network.input
                output = self.shared_network.output

            output = Dense (

                self.output_dim, 
                activation = self.activation,
                kernel_initializer = 'random_normal'

            )(output)
            
            self.model = Model(input,output)
            
            # Configures the model for training
            self.model.compile(
                
                optimizer = SGD(learning_rate = self.lr),
                loss = self.loss
            )
    
    @staticmethod
    def get_network(input, dropout_ratio = 0.1):
        
        output = Dense(256, activation = "sigmoid", kernel_initializer = "random_normal")(input)
        output = BatchNormalization()(output)
        output = Dropout(dropout_ratio)(output)
        output = Dense(128, activation = "sigmoid", kernel_initializer = "random_normal")(output)
        output = BatchNormalization()(output)
        output = Dropout(dropout_ratio)(output)
        output = Dense(64, activation = "sigmoid", kernel_initializer = "random_normal")(output)
        output = BatchNormalization()(output)
        output = Dropout(dropout_ratio)(output)
        output = Dense(32, activation = "sigmoid", kernel_initializer = "random_normal")(output)
        output = BatchNormalization()(output)
        output = Dropout(dropout_ratio)(output)

        return Model(input,output)

    def get_loss(self, input, label):

        x = np.array(input).reshape((-1, self.input_dim))
        return super().get_loss(input=x,label=label)

    def predict(self, input):

        x = np.array(input).reshape((1,self.input_dim))
        return super().predict(input=x)

class LSTM_Network(Neural_Network):

    def __init__(self, num_steps=1, *args, **kwargs):

        super().init(*args, **kwargs)
        with graph.as_default():

            if session is not None:
                set_session(session=session)

            self.num_steps = num_steps
            input = None
            output = None

            if self.shared_network is None:

                input = Input((self.num_setps, self.input_dim))
                output = self.get_network(input=input).output

            else:
                
                input = self.shared_network(input)
                output = self.shared_network(output)

            output = Dense(
                            self.output_dim, 
                            activation=self.activation,
                            kernel_initializer='random_normal'
                     )(output)

            self.model = Model(input,output)
            self.model.compile(
                optimizer=SGD(learning_rate=self.lr),
                loss=self.loss
            )

    @staticmethod    
    def get_network(input):

        output = LSTM(
                        256, 
                        dropout=0.1,
                        return_sequences=True,
                        stateful=False,
                        kernel_initializer='random_normal'
                 )(input)
        output = BatchNormalization()(output)
        output = LSTM(
                        128,
                        dropout=0.1,
                        return_sequences=True,
                        stateful=False,
                        kernel_initializer='random_normal'
                 )(output)
        output = BatchNormalization()(output)
        output = LSTM(
                        64,
                        dropout=0.1,
                        return_sequences=True,
                        stateful=False,
                        kernel_initializer='random_normal'
                 )(output)
        output = BatchNormalization()(output)
        output = LSTM(
                        32,
                        dropout=0.1,
                        stateful=False,
                        kernel_initializer='random_normal'
                 )(output)
        output = BatchNormalization()(output)

        return Model(input,output)

    def get_loss(self, input, label):

        x = np.array(input).reshape((-1, self.num_steps, self.input_dim))
        return super().get_loss(input=x,label=label)

    def predict(self, input):

        x = np.array(input).reshape((1, self.num_steps, self.input_dim))
        return super().predict(input=x)

class CNN(Neural_Network):

    def __init__(self, num_steps=1, *args, **kwargs):

        super().__init__(*args, **kwargs)
        with graph.as_default():
            if session is not None:
                set_session(session=session)

            self.num_steps = num_steps
            input = None
            output = None

            if self.shared_network is None:

                input = Input((self.num_steps, self.input_dim, 1))
                output = self.get_network(input=input).output

            else:

                input = self.shared_network.input
                output = self.shared_network.output

            output = Dense (
                self.output_dim,
                activation=self.activation,
                kernerl_initializer='random_normal'
            )(output)

            self.model = Model(input, output)
            self.model.compile(
                optimizer=SGD(learning_rate=self.lr),
                loss=self.loss
            )

    @staticmethod
    def get_network(input):

        output = Conv2D(
                        256,
                        kernel_size=(1,5),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer='random_normal'
                 )(input)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1,2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(
                        128,
                        kernel_size=(1,5),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer='random_normal'
                 )(input)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1,2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(
                        64,
                        kernel_size=(1,5),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer='random_normal'
                 )(input)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1,2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(
                        32,
                        kernel_size=(1,5),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer='random_normal'
                 )(input)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1,2))(output)
        output = Dropout(0.1)(output)
        output = Flatten()(output)

        return Model(input,output)

    def get_loss(self, input, label):

        x = np.array(input).reshape((-1, self.num_steps, self.input_dim, 1))
        return super().get_loss(input=x,label=label)

    def predict(self, input):

        x = np.array(input).reshape((-1, self.num_steps, self.input_dim, 1))
        return super().predict(input=x)