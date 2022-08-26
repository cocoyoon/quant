from lib2to3.pgen2.pgen import DFAState
import numpy as np
import utils

class Agent:

    """
        Args
            - environment: Chart data
            - min_trading_unit: 최소 트레이딩 단위
            - max_trading_unit: 최대 트레이딩 단위
            - delayed_reward_threshold: 지연 임계 보상

        Constant
            - STATE_DIM: State of Agent -> 주식 보유 비율 & 포트폴리오 가치
            - TRADING_CHARGE: 수수료
            - TRADING_TAX: 거래세
            - ACIONS: Behavior of Agent(Buy or Sell)
                
        Variables
            - portfolio_value = (balance) + (num_stocks * current_price)
    """

    STATE_DIM = 2
    TRADING_CHARGE = 0.00015
    TRADING_TAX = 0.0025
    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2
    ACTIONS = [ACTION_BUY, ACTION_SELL]
    NUM_ACTIONS = len(ACTIONS)

    def __init__(

        self,
        environment,
        min_trading_unit=1,
        max_trading_unit=2,
        delayed_reward_threshold=0.05,
    ):

        self.environment = environment
        self.min_trading_unit = min_trading_unit
        self.max_trading_unit = max_trading_unit
        self.delayed_reward_threshold = delayed_reward_threshold
        self.initial_balance = 0
        self.balance = 0
        self.num_stocks = 0
        self.portfolio_value = 0
        self.portfolio_value_base = 0
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.profitloss = 0
        self.profitloss_base = 0
        self.exploration_base = 0

        # Variables that determine the state of Agent
        self.hold_ratio = 0
        self.portfolio_value_ratio = 0

    def reset(self):

        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.portfolio_value_base = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.hold_ratio = 0
        self.portfolio_value_ratio = 0

    def set_random_eps_base(self):

        self.exploation_base = 0.5 + (np.random.rand()/2)

    def set_balance(self, balance):

        """
            Initialize the balance to get profit-loss
        """

        self.initial_balance = balance

    def get_investement_states(self):

        """ 
            Return
                - hold_ratio: 주식 보유 비율 (현재 보유 주식 수 / 현재 보유할 수 있는 최대 주식 수)
                - portfolio_value_ratio: 포트폴리오 가치 비율 (현재 포트폴리오 가치 / 기준 포트폴리오 가치)
        """

        max_holding_amount = int(self.portfolio_value / self.environment.get_price())

        self.hold_ratio = self.num_stocks / max_holding_amount
        self.portfolio_value_ratio = self.portfolio_value / self.portfolio_value_base

        return (self.hold_ratio, self.portfolio_value_ratio)

    def decide_action(self, value, policy, eps):

        """
            eps: Ratio of exploration -> range(0,1)

            Method Logistics
                - If exploration is True, action is determined randomly
                - Otherwise, action is determined by Neural Network.
        """

        confidence = 0
        prediction = policy

        if prediction is None:

            prediction = value

        if prediction is None:

            # If there is no action by Value/Policy Neural Network, explore !
            print("There is no prediction by Neural Network... Explore! ")
            eps = 1
        else:

            # If policy predict by NN and gives us same probabilty -> ex. (buy: 50 %, sell: 50)
            best_prediction = np.max(policy)
            if (policy == best_prediction).all():

                eps = 1

        if np.random.rand() < eps:

            exploration = True
            if np.random.rand() < self.exploration_base:

                action = self.ACTION_BUY
            else:

                action = np.random.randint(self.NUM_ACTIONS - 1) + 1
        else:

            exploration = False
            action = np.argmax(prediction) # 0 or 1
            confidence = 0.5

            if policy is not None:

                confidence = prediction[action]
            elif value is not None:

                confidence = utils.sigmoid(prediction[action])

            return action, confidence, exploration

    def validate_action(self, action):

        """
            Check whether action is validated.
                - For buying, we can't buy if there is not enough balance.
                - For selling, we can't sell if there is no holding stocks
        """

        if action == Agent.ACTION_BUY:

            min_trading_cost = (1+self.TRADING_CHARGE) * self.min_trading_unit

            if self.balance < (self.environment.get_price() * min_trading_cost):

                return False
        elif action == Agent.ACTION_SELL:

            if self.num_stocks <= 0:

                return False

        return True

    def decide_trading_unit(self, confidence):

        """
            confidence: range(0,1). If it is closer to 1, we added more trading unit.
            Otherwise, no additional_trading.
        """

        trading_unit = self.max_trading_unit - self.min_trading_unit
        if np.isnan(confidence):

            return self.min_trading_unit

        additional_trading_unit = max(min(int(confidence * trading_unit), trading_unit), 0)
        final_trading_unit = self.min_trading_unit + additional_trading_unit
        
        return final_trading_unit

    def act(self, action, confidence):

        """
            Act what agent has decided.
            action: [0: BUY, 1: SELL] decided by NN
        """

        if not self.validate_action(action):

            action = Agent.ACTION_HOLD

        self.immediate_reward = 0
        current_price = self.environment.get_price()
        buy_cost = current_price * (1 + self.TRADING_CHARGE)
        sell_cost = current_price * (1 - (self.TRADING_CHARGE + self.TRADING_TAX))

        # buy
        # trading_unit: at least min_trading_unit
        if action == Agent.ACTION_BUY:

            trading_unit = self.decide_trading_unit(confidence=confidence)
            balance = self.balance - (buy_cost * trading_unit)

            # if it exceeds balance...
            if balance < 0:

                trading_unit = max(
                    min(int(self.balance / buy_cost), self.max_trading_unit),
                    self.min_trading_unit,
                )

            total_buy_cost = buy_cost * trading_unit
            if total_buy_cost > 0:

                self.balance -= total_buy_cost
                self.num_stocks += trading_unit
                self.num_buy += 1

        # sell
        elif action == Agent.ACTION_SELL:

            trading_unit = self.decide_trading_unit(confidence=confidence)
            trading_unit = min(trading_unit, self.num_stocks)

            total_sell_cost = sell_cost * trading_unit
            if total_sell_cost > 0:

                self.balance += total_sell_cost
                self.num_stocks -= trading_unit
                self.num_sell += 1

        # hold
        elif action == Agent.ACTION_HOLD:

            """
                While holding, portfolio value would be changed, so we need to update our portfolio value.
                Here, we return whether our action was good/bad.

                portfolio_value_base: Base of the Portfolio value whenever we reach our delayed_reward_threshold.
                profit_loss_base: Base of the profit-loss when portfolio_value has updated.
            """
            
            self.num_hold += 1

        self.portfolio_value = self.balance + (current_price * self.num_stocks)
        pnl = (self.portfolio_value - self.initial_balance)
        self.profitloss = pnl / self.initial_balance

        self.immediate_reward = self.profitloss
        delayed_reward = 0

        pv_diff = (self.portfolio_value - self.portfolio_value_base)
        self.profitloss_base = pv_diff / self.portfolio_value_base

        if (
            self.profitloss_base > self.delayed_reward_threshold
            or self.profitloss_base < -self.delayed_reward_threshold
        ):

            self.portfolio_value_base = self.portfolio_value
            delayed_reward = self.immediate_reward
        else:

            delayed_reward = 0

        return self.immediate_reward,delayed_reward, 
