
from mplfinance.original_flavor import candlestick_ohlc
from agent import Agent
import threading
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data_manager

lock = threading.Lock()

class Visual_Manager:

    COLORS = ['r', 'b', 'g']

    def __init__(self, vnet=False):

        self.canvas = None
        self.fig = None
        self.axes = None
        self.title = ''

    def chart_canvas(self, chart_data, title):

        """
            Settings for canvas of entire chart
            Day chart is common part regardless of the epoch, I have initialized it here.
        """

        self.title = title
        with lock:

            self.fig, self.axes = plt.subplots (

                nrows=5,
                ncols=1,
                facecolor='w',
                sharex=True
            )

        for ax in self.axes:

            ax.get_xaxis().get_major_formatter().set_scientific(False)
            ax.get_yaxis().get_major_formatter().set_scientific(False)
            ax.yaxis.tick_right()

        # Chart 1 - Day Chart
        print("Plotting Chart 1...")
        self.axes[0].set_ylabel('Env')
        x = np.arange(len(chart_data))
        ohlc = np.hstack(
            (
                x.reshape(-1,1),
                np.array(chart_data)[:, 0:-1]
            )
        )
        candlestick_ohlc (

            ax=self.axes[0],
            quotes=ohlc,
            colorup='r',
            colordown='b'
        )
        ax = self.axes[0].twinx()
        volume = np.array(chart_data)[:,-1].tolist()
        ax.bar(x, volume, color='b', alpha=0.3)

        plt.title(self.title)
        plt.show()

    def plot(
        self,
        num_epoch=None,
        total_epochs=None,
        eps=None,
        action_list=None,
        actions=None,
        num_stocks=None,
        values=[],
        policies=[],
        exps=None,
        lr_idxes=None,
        initial_balance=None,
        pvs=None
    ):

        """
            args:
                - epsilon: exploration rate
                - action_list: What agent can do [Buy, Sell]
                - actions: What agent has done
                - num_stocks: Number of stocks holding
                - values: Output of value by NN
                - policies: Output of policy by NN
                - exps: List of whether explores or not
                - lr_idxes: Learning location
                - pvs: List of portfolio values
        """
        with lock:

            x = np.arange(len(actions))
            actions = np.array(actions)
            values = np.array(values)
            policies = np.array(policies)
            pvs_base = np.zeros(len(actions)) + initial_balance

        # Chart 2 - Status of Agent: Actions & Number of Stocks
        # (('Buy','r'),('Sell','b'))
        print("Plotting Chart 2...")
        for (action, color) in zip(action_list, self.COLORS):
            for idx in x:
                
                if action == actions:
                    
                    self.axes[1].axvline(
                        
                        idx, 
                        color=color, 
                        alpha=0.1
                    )

        # plot number of stocks holding
        self.axes[1].plot(x, num_stocks, '-k')

        print("Plotting Chart 3...")
        # Chart 3 - Value Neural Network
        if len(values) > 0:
            
            max_actions = np.argmax(values, axis=1)
            for (action, color) in zip(action_list, self.COLORS):
                for idx in x:
                    if max_actions[idx] == action:

                        self.axes[2].axvline(
                            
                            idx, 
                            color=color, 
                            alpha=0.1
                        )

                self.axes[2].plot(
                    
                    x, 
                    values[:, action], 
                    color=color, 
                    linestyle='-'
                )

        # Chart 4 - Policy Network 
        # index where explores
        print("Plotting Chart 4...")
        for exp_idx in exps:
            self.axes[3].axvline(exp_idx, color='y')

        _out_vals = policies if len(policies) > 0 else values

        for (idx, out_val) in zip(x, _out_vals):
            
            color = 'white'

            if np.isnan(out_val.max()):
                continue

            elif out_val.argmax() == Agent.ACTION_BUY:
                color = 'r'

            elif out_val.argmax() == Agent.ACTION_SELL:
                color = 'b'

            self.axes[3].axvline(
                
                idx, 
                color=color, 
                alpha=0.1
            )

        # Output of Policy Neural Network
        if len(policies) > 0:
            for (action, color) in zip(action_list, self.COLORS):
                self.axes[3].plot(

                    x,
                    policies[:, action],
                    color=color,
                    linestyle='-'
                )

        # Chart 5 - Portfolio Value
        print("Plotting Chart 5...")
        
        self.axes[4].axhline(

            initial_balance, 
            linestyle='-',
            color='gray'
        )

        # fill 'red' when portfolio value is greater that portfolio_base
        self.axes[4].fill_between(
            
            x, 
            pvs, 
            pvs_base,
            where=(pvs > pvs_base),
            facecolor='r',
            alpha=0.1
        )

        # fill 'blue' when portfolio value is greater that portfolio_base
        self.axes[4].fill_between(
            
            x, 
            pvs, 
            pvs_base,
            where=(pvs < pvs_base),
            facecolor='b',
            alpha=0.1
        )

        self.axes[4].plot(

            x,
            pvs,
            '-k'
        )

        # mark where agent has learned with yellow
        for idx in lr_idxes:
            self.axes[4].axvline(idx, color='y')

        # Ratio of epochs and exploration rate
        self.fig.suptitle("{title} \nEpoch:{num_epoch}/{total_epoch} e={eps}".format(

            title=self.title,
            num_epoch=num_epoch,
            total_epochs=total_epochs,
            eps=eps
        ))
        
        self.fig.title_layout()
        self.fig.subplots_adjust(top=0.85)

    def clear(self, xlim):

        with lock:

            _axes = self.axes.tolist()
            for ax in _axes[1:]:
                
                ax.cla()
                ax.relim()
                ax.autoscale()

            self.axes[1].set_ylabel('Agent')
            self.axes[2].set_ylabel('Value')
            self.axes[3].set_ylabel('Policy')
            self.axes[4].set_ylabel('PV')

            for ax in _axes:

                ax.set_xlim(xlim)
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)
                ax.ticklabel_format(useOffset=False)
    
    def save(self, path):

        """
            save figure on path
        """

        with lock:
            self.fig.savefig(path)

if __name__ == '__main__':

    visual_manager = Visual_Manager()
    df = data_manager.get_data_yahoo('AAPL', start='2021-01-01', end='2022-01-01')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    print(df)
    visual_manager.chart_canvas(chart_data=df, title='Agent Status')