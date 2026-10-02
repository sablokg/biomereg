use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::mean_squared_error;
use smartcore::model_selection::train_test_split;
use smartcore::tree::decision_tree_regressor::DecisionTreeRegressor;
use smartcore::tree::decision_tree_regressor::DecisionTreeRegressorParameters;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
/*
Gaurav Sablok
gsablok@proton.me
*/
pub fn decisiontree(pathfile: &str) -> Result<String, Box<dyn Error>> {
    let fileopen = File::open(pathfile).expect("file not found");
    let fileread = BufReader::new(fileopen);
    let mut datavec: Vec<Vec<f64>> = Vec::new();
    let mut datalabels: Vec<f64> = Vec::new(); // f64, not i32

    for i in fileread.lines() {
        let line = i.expect("line not present");
        let linevec = line.split(",").collect::<Vec<_>>();
        let datalabel = linevec[0..linevec.len() - 1]
            .to_vec()
            .iter()
            .map(|x| x.parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        let label = linevec[linevec.len() - 1..linevec.len()]
            .concat()
            .parse::<f64>() // parse as f64
            .unwrap();
        datavec.push(datalabel);
        datalabels.push(label);
    }

    let datadense = DenseMatrix::from_2d_vec(&datavec).unwrap();
    let trainsplit = train_test_split(&datadense, &datalabels, 0.2, true, Some(4849));
    let trainsplit_0 = trainsplit.0; // x_train
    let trainsplit_1 = trainsplit.1; // x_test
    let trainsplit_2 = trainsplit.2; // y_train
    let trainsplit_3 = trainsplit.3; // y_test

    let params = DecisionTreeRegressor::fit(
        &trainsplit_0,
        &trainsplit_2,
        DecisionTreeRegressorParameters::default()
            .with_min_samples_leaf(2)
            .with_max_depth(4)
            .with_min_samples_split(10),
    )
    .unwrap();

    let predictval = params.predict(&trainsplit_1).unwrap();
    let score = mean_squared_error(&trainsplit_3, &predictval);
    println!("The mean squared error of the model is {}", score);

    Ok("decision tree regressor has been trained".to_string())
}
