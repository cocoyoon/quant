
use cex;
use dex;

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
