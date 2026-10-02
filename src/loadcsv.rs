use std::error::Error;

/*
Gaurav Sablok
gsablok@proton.me
*/

type Vector = (Vec<Vec<f64>>, Vec<f64>, Vec<String>);

pub fn load_csv(path: &str) -> Result<Vector, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let taxa_names: Vec<String> = headers
        .iter()
        .skip(1)
        .take(headers.len() - 2)
        .map(|s| s.to_string())
        .collect();

    let mut features = Vec::new();
    let mut targets = Vec::new();

    for record in reader.records() {
        let record = record?;
        let n = record.len();
        let row: Vec<f64> = (1..n - 1)
            .map(|i| record[i].parse::<f64>().unwrap_or(0.0))
            .collect();
        let y: f64 = record[n - 1].parse()?;
        features.push(row);
        targets.push(y);
    }

    Ok((features, targets, taxa_names))
}
