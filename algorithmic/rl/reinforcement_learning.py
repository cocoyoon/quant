
from json import load
from multiprocessing import pool
import utils 
from environment import Environment
from agent import Agent
from neural_network import Neural_Network, DNN, LSTM_Network, CNN
from visual_manager import Visual_Manager
import os
import logging
import abc
import collections
import threading
import time
import numpy as np
import rl_handler 

class RL(metaclass=abc.ABCMeta):

    lock = threading.Lock() 

    def __init__(

        self,
        rl='rl',
        ticker=None,
        chart=None,
        training_data=None,
        min_trading_unit=1,
        max_trading_unit=2,
        delayed_reward_threshold=0.05,
        nn='dnn',
        num_steps=1,
        learning_rate=0.001,
        value_network=None,
        policy_network=None,
        output_path='',
        load_models=True
    ):  

        """
            args
                - rl: Method of Reinforcement Learning => [DQNLearner, PolicyGradientLearner, ...] 
                - nn: Neural Network => [DNN, LSTM, CNN]
                - load_models: Load pre-trained paramters

            Variables
                - num_features: Number of features to be trained
                - memory_* : Storing learning data
                - output_path: path for logging
                - hyperparams: Hyperparameters for Neural Network
                - epoch_count: number of epochs
                - learning_count: number of mini batch training
                - exploration_count: How many agent does exploration based on exploration rate
                - batch_size: mini batch size

            Modules
                - Envrionment 
                - Agent
                - Neural Network 
                - Visual Manager
        """

        # minimum condition for Reinforcement Learning
        assert min_trading_unit > 0 and \
               max_trading_unit > 0 and \
               max_trading_unit >= min_trading_unit and \
               num_steps > 0 and \
               learning_rate > 0

        self.rl = rl
        self.ticker = ticker
        self.chart = chart

        # Environment Setting
        self.env = Environment(chart_data=chart)

        # Agent Setting
        self.agent = Agent(
            
            environment=self.env,
            min_trading_unit=min_trading_unit,
            max_trading_unit=max_trading_unit,
            delayed_reward_threshold=delayed_reward_threshold
        )

        # Training Data
        self.training_data = training_data
        self.training_sample = None
        self.training_data_idx = -1
        
        # STATE_DIM: Ratio of stock holding + Portfolio Value
        self.num_features = self.agent.STATE_DIM 

        if self.traininig_data is not None:
            
            self.num_features += self.training_data.shape[1]

        # Neural Network Setting
        self.nn = nn
        self.hyperparams = {

            'input_dim': self.num_features,
            'output_dim': self.agent.NUM_ACTIONS,
            'lr': learning_rate,
            'num_steps': self.num_steps
        }
        self.value_network = value_network
        self.policy_network = policy_network
        self.load_models = load_models

        # Visual Manger
        self.visual_manager = Visual_Manager()

        # Memory for storing learning data
        self.memory_training_sample = []
        self.memory_action = [] 
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        
        # Epoch components
        self.loss = 0
        self.epoch_count = 0 
        self.exploration_count = 0
        self.batch_size = 0
        self.learning_count = 0 

        # Path for logging
        self.output_path = output_path

    def init_value_network(
        
        self, 
        shared_network=None,
        activation='linear',
        loss='mse'    
    ):
        # hyperparams: {'input_dim': , 'output_dim': , 'lr': ,'num_steps': }
        self.value_network = rl_handler.init_neural_network(
            
            nn=self.nn,
            hyperparams=self.hyperparams,
            shared_network=shared_network,
            activation=activation,
            loss=loss
        )

        if self.load_models and os.path.exists(self.value_network_path):

            self.value_network.load_weights(path=self.value_network_path)

    def init_policy_network(

        self,
        shared_network=None,
        activation='sigmoid',
        loss='mse'
    ): 

        self.policy_network = rl_handler.init_neural_network(

            nn=self.nn,
            hyperparams=self.hyperparams,
            shared_network=shared_network,
            activation=activation,
            loss=loss
        )

        if self.load_models and os.path.exists(self.policy_network_path):

            self.policy_network.load_weights(path=self.pilicy_network_path)

    def reset(self):

        """
            What it does
                - Whenever completed each epoch, we need to reset the agent.
        """

        self.training_sample = None
        self.training_data_idx = -1

        # reset Environment
        self.env.reset()

        # reset Agent
        self.agent.reset()

        # reset visual manager
        self.visual_manager.clear(xlim=[0, len(self.chart)])

        # reset memory
        self.memory_training_sample = []
        self.momory_action = [] 
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []

        # reset components of epochs 
        self.loss = 0
        self.epoch_count = 0 
        self.exploration_count = 0
        self.batch_size = 0
        self.learning_count = 0 

    def get_training_sample(self):

        """
            Get sample for training our model.
        """
        self.env.observe()

        if len(self.training_data) > (self.training_data_idx + 1):

            self.training_data_idx + 1
            self.training_sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.training_sample.extend(self.agent.get_states())

            return self.training_sample

        return None

    @abc.abstractmethod
    def get_batch(self, batch_size, delayed_reward, discount_factor):

        pass

    def update_networks(self,batch_size, delayed_reward, discount_factor):

        (x, y_value, y_policy) = self.get_batch(batch_size, delayed_reward, discount_factor)

        if len(x) > 0:
            
            loss = 0
        
            if y_value is not None:
                loss += self.value_network.get_loss(input=x, label=y_value)

            if y_policy > 0:
                loss += self.policy_network.get_loss(input=x, label=y_policy)

            return loss
        
        return None

    def fit(self, delayed_reward, discount_factor):
        
        if self.batch_size > 0:

            _loss = self.update_networks(

                batch_size=self.batch_size,
                delayed_reward=delayed_reward,
                discount_factor=discount_factor
            )

            if _loss is not None: 

                self.loss += abs(_loss)
                self.learning_count += 1
                self.memory_learning_idx.append(self.training_data_idx)

            self.batch_size = 0

    def visualize(self, num_epoch, total_epochs, eps):
        
        """
            For CNN, LSTM: our data is about (num_steps -1) less than original, so we put dummy data
            upto (num_steps-1)
        """
        num_dummy = self.hyperparams['num_steps'] - 1
        self.memory_action = [Agent.ACTION_HOLD]*num_dummy + self.memory_action
        self.memory_num_stocks = [0]*num_dummy + self.memory_num_stocks

        if self.value_network is not None:
            
            self.memory_value = [np.array(np.nan)*num_dummy] + self.memory_value

        if self.policy_network is not None:

            self.memory_policy = [np.array(np.nan)*num_dummy] + self.memory_policy

        self.memory_pv = [self.agent.initial_balance]*num_dummy+ self.memory_pv

        self.visual_manager.plot(

            num_epoch=num_epoch,
            total_epochs=total_epochs,
            eps=eps,
            action_list=self.agent.ACTIONS,
            actions=self.memory_action,
            num_stocks=self.memory_num_stocks,
            values=self.memory_value,
            policies=self.memory_policy,
            exps=self.memory_exp_idx,
            lr_idxes=self.memory_learning_idx,
            initial_balance=self.agent.initial_balance,
            pvs=self.memory_pv
        )

    def run(
        
        self,
        total_epochs=100,
        initial_balance=10000,
        discount_factor=0.9,
        start_eps=0.5,
        learning=True
    ):
        """
            Variables
                - max_portfolio_value: Store maximum value of portfolio for entire epochs
                - epoch_profit_count: Store epoch where our agent get profits
        """
        lr = self.hyperparams['lr']
        info = "{ticker} RL: {rl} NN: {nn} Learning_Rate: {lr} "\
               "Discount_Factor: {discount_factor} Trading_Unit: [{min_trading_unit}, " \
               "{max_trading_unit}] Delayed_Reward_Threshold: {delayed_reward_threshold}".format(

                   ticker=self.ticker,
                   rl=self.rl,
                   nn=self.nn,
                   lr=lr,
                   discount_factor=discount_factor,
                   min_trading_unit=self.agent.min_trading_unit,
                   max_trading_unit=self.agent.max_trading_unit,
                   delayed_reward_threshold=self.agent.delayed_reward_threshold
               )

        with self.lock:
            logging.info(msg=info)

        time_start = time.time()

        self.visual_manager.chart_canvas(
            chart_data=self.env.chart_data,
            title=info
        )

        self.epoch_result_dir = os.path.join(

            self.output_path, 'RLTrade_epoch_result_{}'.format(self.ticker)
        )

        if not os.path.isdir(self.epoch_result_dir):

            os.makedirs(self.epoch_result_dir)
        else:

            for file in os.listdir(self.epoch_result_dir):
                
                os.remove(os.path.join(self.epoch_result_dir, file))

        # Initialize agent balance
        self.agent.set_balance(balance=initial_balance)
        max_portfolio_value = 0
        epoch_profit_count = 0

        for epoch in total_epochs:

            time_start_epoch = time.time()
            training_sample_queue = collections.deque(maxlen=self.hyperparams['num_steps']-1)
            self.reset()

            if learning:
                
                epsilon = start_eps * float(1/(epoch/(total_epochs-1)))   
                self.agent.set_random_eps_base()
            else:

                epsilon = start_eps

            while True:

                next_training_sample = self.get_training_sample()
                if next_training_sample is None:

                    break

                training_sample_queue.append(next_training_sample)
                if len(training_sample_queue) < self.hyperparams['num_steps']:

                    continue
                
                pred_value = None
                pred_policy = None

                if self.value_network is not None:

                    pred_value = self.value_network.predict(input=list(training_sample_queue))

                if self.policy_network is not None:

                    pred_policy = self.policy_network.predict(input=list(training_sample_queue))

                (action, confidence, exploration) = self.agent.decide_action(

                    value=pred_value,
                    policy=pred_policy,
                    eps=epsilon
                )

                (immediate_reward, delayed_reward) = self.agent.act(

                    action=action,
                    confidence=confidence
                )

                self.memory_training_sample.append(list(training_sample_queue))
                self.memory_action.append(action)
                self.memory_reward.append(immediate_reward)
                
                if self.value_network is not None:

                    self.memory_value.append(pred_value)

                if self.policy_network is not None:

                    self.memory_policy.append(pred_policy)
                
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                
                if exploration: 

                    self.memory_exp_idx.append(self.training_data_idx)
                    self.exploration_count += 1

                self.batch_size += 1
                self.epoch_count += 1

                if learning and (delayed_reward != 0):

                    self.fit(

                        delayed_reward=delayed_reward,
                        discount_factor=discount_factor
                    )


        

                