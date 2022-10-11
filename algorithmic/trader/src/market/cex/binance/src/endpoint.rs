
pub enum EndPoint {
    Base,
    Spot,
    Future,
    MarketData(MarketData),
    UserData,
}

pub enum MarketData {
    ExhchangeInfo,
    Candle,
    PriceChangeFor24HR,
    SymbolPrice, 
}

impl From<EndPoint> for String {
    fn from(end_point: EndPoint) -> Self {
        match end_point {
            EndPoint::Base => String::from("https://api.binance.com"),
            EndPoint::Spot => "".into(),
            EndPoint::Future => "".into(),
            EndPoint::MarketData(market_data) => {
                match market_data {
                    MarketData::ExhchangeInfo => String::from("/api/v3/exchangeInfo"),
                    MarketData::Candle => String::from("/api/v3/klines"),
                    MarketData::PriceChangeFor24HR => String::from("/api/v3/ticker/24hr"),
                    MarketData::SymbolPrice => String::from("/api/v3/ticker/price"),
                }
            }
            EndPoint::UserData => "".into(),
        }
    }
} 



