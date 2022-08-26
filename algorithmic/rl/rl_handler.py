
from neural_network import DNN, LSTM_Network, CNN

def init_neural_network(nn, hyperparams, shared_network, activation, loss):

    """
        Args
            - nn: Neural Network
            - hyperparams: {'input_dim': , 'output_dim': , 'lr': ,'num_steps': }
            - shared_network: Other network 
            - activation: 'linear' or 'sigmoid'
            - loss: loss function

        Return 
            - rl_network(Value/Policy): DNN | CNN | LSTM 
    """

    rl_network = None

    if nn == 'dnn':

        rl_network = DNN(

            input_dim=hyperparams['input_dim'],
            output_dim=hyperparams['output_dim'],
            lr=hyperparams['lr'],
            shared_network=shared_network,
            activation=activation,
            loss=loss
        )

    elif nn == 'lstm':

        rl_network = LSTM_Network(

            input_dim=hyperparams['input_dim'],
            output_dim=hyperparams['output_dim'],
            lr=hyperparams['lr'],
            num_steps=hyperparams['num_steps'],
            shared_network=shared_network,
            activation=activation,
            loss=loss
        )

    elif nn == 'cnn':

        rl_network = CNN(

            input_dim=hyperparams['input_dim'],
            output_dim=hyperparams['output_dim'],
            lr=hyperparams['lr'],
            num_steps=hyperparams['num_steps'],
            shared_network=shared_network,
            activation=activation,
            loss=loss
        )

    return rl_network