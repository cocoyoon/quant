import api_key as key
import pyupbit as upbit
import rsi as get_rsi
import bolinger
import read_data as read
import time
import base_model as base 

def main():

    coin_list = read.coin_list()
    interval = "day"
    count = 200
    to = None

    for coin in coin_list: 

        time.sleep(1)
        df = read.coin_data(coin, interval, count, to)
        last_index = len(df)-1
        
        try:
            rsi = get_rsi.relative_strength(14, count, df)
            bolinger_df = bolinger.calculate_band(20, df)

            date_time = str(df.index[last_index])

            change = base.get_diff(df, date_time)
            price_change = change[0]
            volume_change = change[1]

        except Exception as e:

            print(e)
            continue

        print(coin, "->" , base.price(coin), "KRW / " \
            "Price Change:", price_change,"% ", "Volume_Change:", volume_change,"%")
        
        print("RSI: ", rsi)
        base.obv(df,13)

        if bolinger.signal(bolinger_df) == "sell" and rsi > 70:

            print("SELL SIGNAL!", coin)

        elif bolinger.signal(bolinger_df) == "buy" and rsi < 30:

            print("BUY SIGNAL!", coin)

if __name__ == '__main__':

    main()