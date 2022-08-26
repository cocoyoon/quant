from datetime import datetime as date
from dateutil.relativedelta import relativedelta as before


class Utils:

    @staticmethod
    def get_date(period=None) -> list :

        """
        Return
        - [1 year before today , date of today] 
        """
        _today = date.now()
        _from = _today - before(years=1)
        
        return [_from.strftime('%Y%m%d'), _today.strftime('%Y%m%d')]

    @staticmethod
    def pct_change(price_1, price_2) -> float:

        if price_1 == 0:
            return 0
        
        return (price_2 - price_1) / price_1 * 100
        

if __name__ == "__main__":

    print(Utils.calculate_date())
    
