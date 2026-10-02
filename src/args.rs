use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "biomereg",
    version = "1.0",
    about = "Regression Modelling for microbiome
       ************************************************
       Gaurav Sablok,
       Email: gsablok@proton.me
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Random modelling regression
    Random {
        /// path to the filename
        pathname: String,
        /// Threads for the analysis
        thread: String,
    },
    /// Logistic modelling regression
    Logistic {
        /// path to the filename
        pathname: String,
        /// Threads for the analysis
        thread: String,
    },
    /// KNN modelling regression
    Knn {
        /// path to the filename
        pathname: String,
        /// Threads for the analysis
        thread: String,
    },
    /// XGB modelling regression
    XGBoost {
        /// path to the filename
        pathname: String,
        /// Threads for the analysis
        thread: String,
    },
    /// tsne modelling regression
    TSNE {
        /// path to the filename
        pathname: String,
        /// Threads for the analysis
        thread: String,
    },
    /// Decision modelling regression
    DecisionTree {
        /// path to the filename
        pathname: String,
        /// Threads for the analysis
        thread: String,
    },
}
