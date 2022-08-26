from time import time
import pyupbit as upbit
from datetime import datetime,timedelta

def price(ticker):

    """
        ticker: "KRW-BTC", "KRW-XRP" ...
    """
    try:
        price = upbit.get_current_price(ticker)

        if price is None:

            raise Exception("No Price Data Returned")

        return int(price)

    except Exception as e:

        print(e)

def get_diff(df, date_time):

    """
        전일 대비 거래량 차이 Percentage
        
        Parameters

        date_time: Defalut: 'today' 09:00:00. 
        You can change whatever you want with specific format

        Return

        Tuple(price difference, volume difference)
    """

    try:
        
        date_index = binary_search(0,len(df)-1,date_time,df)

        if date_index == -1:

            raise Exception("There is no such Date in current DataFrame")

        search_date = df.index[date_index]
        search_date_one_day_before = df.index[date_index-1]

        df_diff = df.copy()[['close','volume']].diff(1)

        volume_diff_percentage = (df_diff.loc[search_date]['volume'] / df.loc[search_date_one_day_before]['volume']) * 100
        price_diff_percentage = (df_diff.loc[search_date]['close'] / df.loc[search_date_one_day_before]['close']) * 100

        return formatter(price_diff_percentage),formatter(volume_diff_percentage)

    except Exception as e: 

        print(e)

def obv(df,time_span):

    obv_df = df.copy()[['close','volume']]
    obv_list = get_obv(obv_df)

    obv_df['obv'] = obv_list
    obv_df['obv_ema'] = obv_df['obv'].ewm(com = time_span,adjust = False).mean()

    if obv_df['obv'][-1] > obv_df['obv_ema'][-1]:

        print("BULLISH!")

    else:
    
        print("BEARISH!")

def get_obv(df):

    obv_list = []
    obv_list.append(0)

    for i in range(1,len(df)):

        if df['close'][i] > df['close'][i-1]:

            volume = obv_list[-1] + df['volume'][i]
            obv_list.append(volume)

        elif df['close'][i] < df['close'][i-1]:

            volume = obv_list[-1] - df['volume'][i]
            obv_list.append(volume)

        else: 

            volume = obv_list[-1]
            obv_list.append(volume)

    return obv_list

def formatter(num):

    """
        format float upto 2 decimal number
    """

    return float(format(num,".2f"))

def binary_search(left,right,date_time,df):

    while left <= right: 

        mid = int(left + (right - left) / 2)

        if str(df.index[mid]) == date_time:

            return mid

        if str(df.index[mid]) > date_time:

            right = mid - 1

        else:

            left = mid + 1

    return -1


if __name__ == "__main__":

    df = upbit.get_ohlcv("KRW-BTC","day",200,None)

    print(get_diff(df,"2021-09-28 09:00:00"))

    obv(df,13)
    