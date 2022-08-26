import read_data as data

"""
    Bolinger Band 

    Get data from 'read_data.py' with certain parameters.

    Calculate Standard Diviation.

    Catch the buy/sell signal by certain condition.
"""

def calculate_band(time_period,df):

    """
        sigma: Standard Diviation

        Upper Band: mean price + 2(sigma)

        Bottom Band: mean price - 2(sigma)
    """

    bolinger_df = df[['close']].copy()

    bolinger_df['middle'] =  bolinger_df['close'].rolling(time_period).mean()

    bolinger_df['upper'] = bolinger_df['close'].rolling(time_period).mean() \
            + (2 * (bolinger_df['close'].rolling(time_period).std()))

    bolinger_df['bottom'] = bolinger_df['close'].rolling(time_period).mean() \
            - (2 * (bolinger_df['close'].rolling(time_period).std()))

    return bolinger_df

def signal(bolinger_df):
    
    """
        count: number of data in df 

        df: dataframe returned from calculate_band()
    """

    current_price = bolinger_df['close'][len(bolinger_df)-1]
    upper_band = bolinger_df['upper'][len(bolinger_df)-1]
    bottom_band = bolinger_df['bottom'][len(bolinger_df)-1]

    if current_price > upper_band: 

        return "sell"

    elif current_price <= bottom_band:

        return "buy"

    else: 

        return "nothing"

# def signal(self):

#     """
#         if current price is on upper band, 'sell' signal

#         elif current price is on below bottom band, 'buy' singal

#         else no signal 
#     """

#     for i in range(len(self.df['close'])):

#         if self.df['close'][i] > self.df['upper'][i]:

#             self.buy.append(np.nan)
#             self.sell.append(self.df['close'][i])

#         elif self.df['close'][i] < self.df['bottom'][i]:

#             self.buy.append(self.df['close'][i])
#             self.sell.append(np.nan)

#         else:

#             self.buy.append(np.nan)
#             self.sell.append(np.nan)

#     # Put buy/sell data into our data frame
#     self.df['buy'] = self.buy
#     self.df['sell'] = self.sell

#     # Eliminate the null data
#     self.buy_data = self.df.loc[self.df['buy'].notnull()]
#     self.sell_data = self.df.loc[self.df['sell'].notnull()]

#     # create 'signal' data frame 
#     self.signal = pd.concat([self.buy_data,self.sell_data])
#     self.signal = self.signal.reset_index(drop = True)

# def plot(self):

#     plt.figure(figsize=(30,15.5))

#     graph = sns.lineplot(data = self.df, x = self.df.index, y = 'close')
#     sns.lineplot(data = self.df, x = self.df.index , y = 'upper')
#     sns.lineplot(data=  self.df, x = self.df.index , y = 'middle')
#     sns.lineplot(data=  self.df, x = self.df.index , y = 'bottom')
#     plt.legend(['Close Price', 'Upper Band', 'Middle Band', 'Lower Bound'])
#     graph.set_title("Bollinger Bands:" + self.coin_ticker)

#     plt.scatter(self.df.index, self.df['buy'],label = 'Buy', marker = '^',s = 200,color = 'green', alpha = 1)
#     plt.scatter(self.df.index, self.df['sell'],label = 'Sell', marker = 'v',s = 200, color = 'red', alpha = 1)



if __name__ == "__main__":

    coin_data = data.coin_data("KRW-BTC","minute1",200,None)
    df = calculate_band(20,coin_data)
    signal(200,df)

    
    


    

