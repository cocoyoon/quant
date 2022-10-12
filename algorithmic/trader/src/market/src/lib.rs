pub use binance::{
    BinanceClient, 
    EndPoint,
    binance::get_market_data,
};

pub enum Exchange {
    Cex(CexType),
    Dex(DexType),
}

pub enum CexType {
    Binance,
    Ftx,
}

pub enum DexType {
    UniSwap,
}
