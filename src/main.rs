use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use tpue::cli::{Cli, Command};
use tpue::config::Config;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_new(&cli.log_level)?)
        .init();
    let config = Config::load(cli.config_path())?;
    match cli.command {
        Command::Serve => tpue::server::serve(config).await,
        Command::Detect { image } => {
            let response = tpue::server::detect_file(config, &image).await?;
            println!("{}", serde_json::to_string(&response)?);
            Ok(())
        }
        Command::Models => {
            println!(
                "{}",
                serde_json::to_string(&tpue::server::models_json(&config))?
            );
            Ok(())
        }
    }
}
