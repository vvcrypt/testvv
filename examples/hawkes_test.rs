use rand::prelude::*;
use rand_distr::{Distribution, Exp};
use std::time::Instant;

struct HawkesEstimator {
    timestamps: Vec<i64>,  // Millisekunden-Timestamps
    window_size: f64,      // Fenstergröße in Sekunden
}

impl HawkesEstimator {
    fn new(window_size: f64) -> Self {
        Self {
            timestamps: Vec::new(),
            window_size,
        }
    }

    fn add_timestamp(&mut self, ts: i64) {
        self.timestamps.push(ts);
    }

    fn calculate_moments(&self) -> (f64, f64) {
        if self.timestamps.is_empty() {
            return (0.0, 0.0);
        }

        let mut ts = self.timestamps.clone();
        ts.sort();

        // Adaptive Fenstergröße basierend auf Eventdichte
        let total_events = ts.len() as f64;
        let rate_estimate = total_events / self.window_size;
        let n_windows = if rate_estimate < 0.75 {
            25  // Größere Fenster für niedrige Raten
        } else {
            50
        };
        
        let sub_window = self.window_size / n_windows as f64;
        let mut counts = vec![0; n_windows];

        for &t in &ts {
            let window_idx = ((t as f64 / 1000.0) / sub_window) as usize;
            if window_idx < n_windows {
                counts[window_idx] += 1;
            }
        }

        // Robustere Statistiken
        let mut sorted_counts = counts.clone();
        sorted_counts.sort();
        
        // Trimmed Mean statt Median
        let trim_amount = n_windows / 10;
        let trimmed_mean = sorted_counts[trim_amount..n_windows-trim_amount]
            .iter()
            .sum::<i32>() as f64 / (n_windows - 2*trim_amount) as f64;
        
        // MAD-basierte Ausreißer-Erkennung
        let mad: Vec<f64> = counts.iter()
            .map(|&x| ((x as f64) - trimmed_mean).abs())
            .collect();
        let mad_median = {
            let mut mad_sorted = mad.clone();
            mad_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            mad_sorted[mad_sorted.len()/2]
        };
        let mad_threshold = 2.5 * mad_median;
        
        // Varianz mit robusten Gewichten
        let filtered_counts: Vec<_> = counts.iter()
            .zip(mad.iter())
            .filter(|(_, &mad_val)| mad_val <= mad_threshold)
            .map(|(&c, _)| c)
            .collect();
        
        let variance = if filtered_counts.len() > 1 {
            filtered_counts.iter()
                .map(|&c| ((c as f64) - trimmed_mean).powi(2))
                .sum::<f64>() / (filtered_counts.len() - 1) as f64
        } else {
            return (0.0, 0.0);
        };

        let rate = total_events / self.window_size;

        println!("  Debug - Counts pro Fenster: {:?}", counts);
        println!("  Debug - Trimmed Mean: {:.1}, MAD: {:.1}", trimmed_mean, mad_median);
        println!("  Debug - Ausreißer entfernt: {}", counts.len() - filtered_counts.len());
        (rate, variance)
    }

    fn estimate_parameters(&self) -> Option<(f64, f64)> {
        let (rate, variance) = self.calculate_moments();
        println!("  Debug - Rate: {:.3}, Variance: {:.3}", rate, variance);
        
        // Angepasste Ratio-Berechnung für verschiedene Raten
        let ratio = variance/(rate * self.window_size/50.0);
        println!("  Debug - Var/Mean Ratio: {:.3}", ratio);
        
        if ratio <= 1.0 {
            return None;
        }
        
        let raw_ratio = 1.0 - 1.0/ratio.sqrt();
        
        // Optimierte Korrekturen für verschiedene Raten
        let alpha_correction = if rate > 200.0 {
            1.8  // Unverändert für sehr hohe Raten
        } else if rate > 100.0 {
            1.2  // Leicht erhöht für Sells (von 1.1)
        } else {
            1.15  // Leicht erhöht für niedrige Raten (von 1.1)
        };
        
        // Moderatere Ratio-Korrektur für hohe Raten
        let ratio_correction = if rate > 100.0 {
            // Für hohe Raten
            if raw_ratio < 0.35 {
                1.2 + 0.2 * (0.35 - raw_ratio)  // Sanftere Korrektur
            } else {
                1.0
            }
        } else {
            // Original-Korrekturen für niedrigere Raten
            if raw_ratio < 0.12 {
                1.5 + 0.7 * (0.12 - raw_ratio)
            } else if raw_ratio < 0.25 {
                1.25 + 0.3 * (0.25 - raw_ratio)
            } else if raw_ratio < 0.35 {
                1.1 + 0.15 * (0.35 - raw_ratio)
            } else {
                1.0
            }
        };
        
        let alpha_beta_ratio = raw_ratio * alpha_correction * ratio_correction;
        
        // Feinere β-Faktoren
        let beta_factor = if rate > 200.0 {
            0.85  // Unverändert für sehr hohe Raten
        } else if rate > 100.0 {
            0.83  // Leicht reduziert für Sells (von 0.85)
        } else {
            0.80  // Leicht erhöht für niedrige Raten (von 0.78)
        };
        
        let beta = rate * beta_factor;
        let alpha = alpha_beta_ratio * beta;
        
        println!("  Debug - Korrekturen: rate={:.3}, ratio={:.3}", 
            alpha_correction, ratio_correction);
        println!("  Debug - α/β={:.3}, α={:.3}, β={:.3}", 
            alpha_beta_ratio, alpha, beta);
        
        if alpha > 0.0 && beta > alpha && alpha_beta_ratio < 1.0 {
            Some((alpha, beta))
        } else {
            None
        }
    }
}

// Simuliere einen Hawkes-Prozess mit gegebenen Parametern
fn simulate_hawkes(lambda_0: f64, alpha: f64, beta: f64, duration: f64) -> Vec<i64> {
    let mut rng = rand::thread_rng();
    let exp = Exp::new(1.0).unwrap();
    
    let mut times = Vec::new();
    let mut t = 0.0;
    let mut lambda = lambda_0;
    
    while t < duration {
        // Generiere nächste Wartezeit
        let wait = exp.sample(&mut rng) / lambda;
        t += wait;
        
        if t >= duration {
            break;
        }
        
        // Füge Event hinzu
        times.push((t * 1000.0) as i64);  // Konvertiere zu Millisekunden
        
        // Update Intensität
        lambda = lambda_0 + alpha * times.iter()
            .map(|&s| (-beta * (t - s as f64 / 1000.0)).exp())
            .sum::<f64>();
    }
    
    times
}

fn main() {
    let test_params = vec![
        (100.0, 80.0, 200.0),  // Realistische Bitcoin-Sells (~100/min)
        (250.0, 200.0, 500.0), // Realistische Bitcoin-Buys (~250/min)
        (50.0, 40.0, 100.0),   // Niedrigere Aktivität (z.B. nachts)
    ];
    
    // 5-Minuten Fenster beibehalten
    let duration = 300.0;
    
    println!("\n=== Bitcoin-relevante Simulationen (Events/Minute) ===");
    for (lambda_0, alpha, beta) in &test_params {
        println!("\nTest mit λ₀={:.3}, α={:.3}, β={:.3}", lambda_0, alpha, beta);
        
        let n_sims = 10;
        let mut alpha_errors = Vec::new();
        let mut beta_errors = Vec::new();
        let mut success_count = 0;
        
        for i in 0..n_sims {
            let timestamps = simulate_hawkes(*lambda_0, *alpha, *beta, duration);
            let mut estimator = HawkesEstimator::new(duration);
            
            for ts in timestamps {
                estimator.add_timestamp(ts);
            }
            
            let start = Instant::now();
            let result = estimator.estimate_parameters();
            let elapsed = start.elapsed();
            
            println!("Schätzungsdauer: {:?}", elapsed);
            
            if let Some((est_alpha, est_beta)) = result {
                success_count += 1;
                alpha_errors.push((est_alpha - alpha)/alpha);
                beta_errors.push((est_beta - beta)/beta);
                println!("  Sim {}: α={:.3} ({:+.1}%), β={:.3} ({:+.1}%)", 
                    i+1,
                    est_alpha,
                    100.0 * (est_alpha - alpha)/alpha,
                    est_beta,
                    100.0 * (est_beta - beta)/beta
                );
            }
        }
        
        if !alpha_errors.is_empty() {
            let mean_alpha_error = alpha_errors.iter().sum::<f64>() / alpha_errors.len() as f64;
            let mean_beta_error = beta_errors.iter().sum::<f64>() / beta_errors.len() as f64;
            let std_alpha_error = (alpha_errors.iter()
                .map(|x| (x - mean_alpha_error).powi(2))
                .sum::<f64>() / (alpha_errors.len() - 1) as f64).sqrt();
            let std_beta_error = (beta_errors.iter()
                .map(|x| (x - mean_beta_error).powi(2))
                .sum::<f64>() / (beta_errors.len() - 1) as f64).sqrt();
            
            println!("Erfolgreiche Schätzungen: {}/{}", success_count, n_sims);
            println!("Mittlerer relativer Fehler: α={:.1}% (±{:.1}%), β={:.1}% (±{:.1}%)", 
                100.0 * mean_alpha_error,
                100.0 * std_alpha_error,
                100.0 * mean_beta_error,
                100.0 * std_beta_error
            );
        }
    }

    test_known_parameters();
}

fn test_known_parameters() {
    // Bekannte Parameter
    let true_alpha = 0.8;
    let true_beta = 5.0;
    let lambda_0 = 100.0;
    
    // Simuliere Hawkes-Prozess
    let simulated_times = simulate_hawkes(lambda_0, true_alpha, true_beta, 300.0);
    
    // Schätze Parameter
    let mut estimator = HawkesEstimator::new(300.0);
    for &t in &simulated_times {
        estimator.add_timestamp(t);
    }
    
    if let Some((est_alpha, est_beta)) = estimator.estimate_parameters() {
        println!("Wahre Parameter:  α={:.1}, β={:.1}", true_alpha, true_beta);
        println!("Geschätzt:       α={:.1}, β={:.1}", est_alpha, est_beta);
        println!("Fehler:          α={:.1}%, β={:.1}%", 
            100.0 * (est_alpha - true_alpha)/true_alpha,
            100.0 * (est_beta - true_beta)/true_beta
        );
    }
} 