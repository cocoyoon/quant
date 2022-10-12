use traderlib::{BinanceClient, get_market_data};

#[tokio::main]
async fn main() -> color_eyre::Result<()>{

    
    get_market_data().await?;

    println!("Hi this is main!");

    Ok(())
}