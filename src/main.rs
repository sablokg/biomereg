mod args;
mod clr;
mod loadcsv;
mod permutate;
mod random;
use self::random::randomforest;
use crate::args::CommandParse;
use crate::args::Commands;
use clap::Parser;
mod knn;
mod logistic;
use crate::knn::knnregressor;
use crate::logistic::logisticreg;
mod xgb;
use crate::xgb::xgbreg;
mod bhtsne;
use crate::bhtsne::pcs_tsne;
mod decision;
use crate::decision::decisiontree;
use vornix_banner::{Banner, BuiltinFont, Style, rgb};

/*
Gaurav Sablok
gsablok@proton.me
*/

#[tokio::main]
async fn main() {
    let style = Style::new().fg(rgb(255, 100, 20)).bold();

    let mut banner = Banner::new("biomeREG")
        .with_builtin_font(BuiltinFont::Block)
        .with_style(style)
        .centered(true);

    banner.display().unwrap();
    let args = CommandParse::parse();
    match &args.command {
        Commands::Random { pathname, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("Threads failed to launch");
            value.install(|| {
                let command = randomforest(pathname).unwrap();

                println!(
                    "The command has finished and the neural logistic has been trained:{:?}",
                    command
                );
            });
        }
        Commands::Logistic { pathname, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("Threads failed to launch");
            value.install(|| {
                let command = logisticreg(pathname).unwrap();

                println!(
                    "The command has finished and the neural logistic has been trained:{:?}",
                    command
                );
            });
        }
        Commands::Knn { pathname, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("Threads failed to launch");
            value.install(|| {
                let command = knnregressor(pathname).unwrap();

                println!(
                    "The command has finished and the neural logistic has been trained:{:?}",
                    command
                );
            });
        }
        Commands::XGBoost { pathname, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("Threads failed to launch");
            value.install(|| {
                let command = xgbreg(pathname).unwrap();

                println!(
                    "The command has finished and the neural logistic has been trained:{:?}",
                    command
                );
            });
        }
        Commands::TSNE { pathname, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("Threads failed to launch");
            value.install(|| {
                let command = pcs_tsne(pathname).unwrap();

                println!(
                    "The command has finished and the neural logistic has been trained:{:?}",
                    command
                );
            });
        }
        Commands::DecisionTree { pathname, thread } => {
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .expect("Threads failed to launch");
            value.install(|| {
                let command = decisiontree(pathname).unwrap();

                println!(
                    "The command has finished and the neural logistic has been trained:{:?}",
                    command
                );
            });
        }
    }
}
