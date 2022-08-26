import read_data as data

def relative_strength(time_period,count,df): 
    
    """
        'time_period' : period for RSI. Default is 14

        'count' : number of df

        'df' : dataframe for close data frame with difference one
    """

    df_diff = df[['close']].diff(1) 

    up_trend = df_diff.copy()
    down_trend = df_diff.copy()
    up_trend = up_trend.clip(lower=0)
    down_trend = (-1)*down_trend.clip(upper=0)

    ema_up = up_trend.ewm(com = time_period-1,adjust = False).mean()
    ema_down = down_trend.ewm(com = time_period-1,adjust = False).mean()

    return current_rsi(ema_up,ema_down,len(df)-1)

def current_rsi(ema_up,ema_down,count):

    rs = ema_up/ema_down

    rsi = 100.0 - (100.0/(1.0 + rs))

    return rsi['close'][count]


if __name__ == "__main__":

    coin_data = data.coin_data("KRW-BTC","day",200,None)
    print(relative_strength(14,200,coin_data))




# def plot(self,current_rsi): 

#    fig,axes = plt.subplots(nrows = 2, figsize = (10,10))

#    sns.lineplot(x = self.rsi_data.index, y =  self.rsi_data['close'], ax = axes[0])
#    sns.lineplot(x = self.df.index, y = self.df['close'], ax = axes[1])

#    axes[0].set_title("RSI: " + self.ticker)
#    axes[0].set_ylabel("RSI")
#    axes[0].axhline(30, color = 'green')
#    axes[0].axhline(70, color = 'green')
#    axes[0].axhline(20, color = 'red')
#    axes[0].axhline(80, color = 'red')

#    axes[1].set_title("Price")

#  #If greater than 70, display "Sell"
#    if current_rsi > 70:
       
#        for x,y in zip(current_data.index,current_data.values):

#             label = "Sell"

#             plt.annotate (label, # this is the text
#                    (x,y), # this is the point to label
#                    textcoords = "offset points", # how to position the text
#                    xytext = (0,10), # distance from text to points (x,y)
#                    ha = 'center',
#                   fontsize = 25) # horizontal alignment can be left, right or center

#             plt.scatter(current_data.index, current_data.values, \
#           label = 'Sell', marker = 'v', s = 200, color = 'red', alpha = 1) 
    
#     elif current_rsi < 30:
        
#         for x,y in zip(current_data.index,current_data.values):

#             label = "Buy"

#             plt.annotate(label, # this is the text
#                    (x,y), # this is the point to label
#                      textcoords = "offset points", # how to position the text
#                   xytext = (0,10), # distance from text to points (x,y)
#                      ha = 'center',
#                     fontsize = 25) # horizontal alignment can be left, right or center
                            
#             plt.scatter(current_data.index, current_data.values,\
#                 label = 'Buy',marker = '^', s = 200, color = 'green', alpha = 1) #plot scatter on RSI plot  
