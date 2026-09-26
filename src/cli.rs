use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "tpue", version, about = "Coral Edge TPU detection service")]
pub struct Cli {
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,
    #[arg(long, global = true, default_value = "info")]
    pub log_level: String,
    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    pub fn config_path(&self) -> PathBuf {
        self.config
            .clone()
            .or_else(|| std::env::var_os("TPUE_CONFIG").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("tpue.toml"))
    }
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Serve,
    Detect { image: PathBuf },
    Models,
}
