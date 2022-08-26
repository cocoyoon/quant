
from traceback import format_exception
from h11 import Data
from matplotlib.pyplot import close
from pykrx import stock as krx
from utils import Utils
import pandas as pd

class Data_Manager:


    @staticmethod
    def get_total_num_listed(market: str) -> int:

        return len(krx.get_market_ticker_list(market=market))

    @staticmethod
    def get_chart_data(ticker: str, freq: str, type:str):

        """
            Args
            - ticker: stock tickers
            - freq: 'd', 'm' , 'y'
            - type: 'ohlcv' or 'market_cap'
        """
        
        (_from,_to) = Utils.get_date()
        if type == 'ohlcv':

            return krx.get_market_ohlcv_by_date(
                fromdate=_from,
                todate=_to,
                ticker=ticker,
                freq=freq
            )
        elif type == 'market_cap':

            return krx.get_market_cap_by_date(
                _from, 
                _to, 
                ticker=ticker, 
                freq=freq
            )

    @staticmethod
    def get_ticker(date: str = None, market: str = "KOSPI") -> list:

        """
            Args
            - date: If not passed, default would be today's date
            - market: default="KOSPI" (KOSPI | KOSDAQ | KONEX | ALL)

            Returns
            - List of Ticker
        """

        return krx.get_market_ticker_list(date=date, market=market)

    @staticmethod
    def get_ticker_name(ticker: str) -> str :

        return krx.get_market_ticker_name(ticker=ticker)

    @staticmethod
    def get_transaction_amount(ticker=None, freq='d') -> list :
        
        df = Data_Manager.get_chart_data(
            ticker=ticker,
            freq=freq,
            type='market_cap'
        )

        return df['거래대금'].values.tolist()

    @staticmethod 
    def get_return_pct_change(ticker: str = None, freq: str ='d', periods=1) -> list:

        df = Data_Manager.get_chart_data(
            ticker=ticker, 
            freq=freq,
            type='ohlcv'
        )

        return df['종가'].pct_change(periods=periods).dropna().mul(100).sort_values().tolist()

    @staticmethod
    def get_tail_change(ticker: str = None, freq: str = 'd') -> list:
        
        df = Data_Manager.get_chart_data(
            ticker=ticker,
            freq=freq,
            type='ohlcv'
        )
        # 1. 양봉인 인덱스 확인
        index = []
        for i in range(len(df)):
            if df['시가'][i] < df['종가'][i]:
                index.append(i)

        # 2. 양봉 긴꼬리 확인
        diff = []
        for i in index:

            price_diff = df['고가'][i] - df['종가'][i]
            peak_price = df['고가'][i]
            if peak_price != 0:
                diff.append((price_diff/peak_price*100))
    
        return sorted(diff)


if __name__ == "__main__":
    
    # print(Data_Manager().get_ticker())
    df = Data_Manager.get_tail_change(ticker='005930')
    print(df)

    # 1. 양봉인 인덱스 확인
    index = []
    for i in range(len(df)):
        if df['시가'][i] < df['종가'][i]:
            index.append(i)

    print(index)
    diff = []
    for i in index:

        diff.append(((df['고가'][i]- df['종가'][i])/df['고가'][i]*100))
    print(sorted(diff))
    print('hi')
    # 2. 긴꼬리 여부 체크
