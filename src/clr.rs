/*
Gaurav Sablok
gsablok@proton.me
*/

pub fn clr_transform(data: &[Vec<f64>], pseudocount: f64) -> Vec<Vec<f64>> {
    data.iter()
        .map(|row| {
            let logs: Vec<f64> = row.iter().map(|&v| (v + pseudocount).ln()).collect();
            let mean_log = logs.iter().sum::<f64>() / logs.len() as f64;
            logs.iter().map(|&l| l - mean_log).collect()
        })
        .collect()
}

pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let mean_y = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ss_tot: f64 = y_true.iter().map(|&y| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&y, &yh)| (y - yh).powi(2))
        .sum();
    1.0 - ss_res / ss_tot
}
