use std::time::Instant;
use rand::prelude::*;
use rand_distr::{Distribution, Exp};
use std::collections::VecDeque;
use serde::Deserialize;

struct HawkesEstimator {
    timestamps: VecDeque<i64>,
    window_size: f64,
    capacity: usize,
}

impl HawkesEstimator {
    fn new(window_size: f64) -> Self {
        let capacity = (window_size * 1000.0) as usize;
        Self {
            timestamps: VecDeque::with_capacity(capacity),
            window_size,
            capacity,
        }
    }

    fn add_timestamp(&mut self, ts: i64) {
        let cutoff = ts - (self.window_size * 1000.0) as i64;
        while let Some(&oldest) = self.timestamps.front() {
            if oldest < cutoff {
                self.timestamps.pop_front();
            } else {
                break;
            }
        }
        self.timestamps.push_back(ts);
    }

    fn calculate_moments(&self) -> (f64, f64) {
        if self.timestamps.is_empty() {
            return (0.0, 0.0);
        }

        let mut ts: Vec<_> = self.timestamps.iter().copied().collect();
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
        
        let trim_amount = n_windows / 10;
        let trimmed_mean = sorted_counts[trim_amount..n_windows-trim_amount]
            .iter()
            .sum::<i32>() as f64 / (n_windows - 2*trim_amount) as f64;
        
        let mad: Vec<f64> = counts.iter()
            .map(|&x| ((x as f64) - trimmed_mean).abs())
            .collect();
        let mad_median = {
            let mut mad_sorted = mad.clone();
            mad_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            mad_sorted[mad_sorted.len()/2]
        };
        
        let mad_threshold = 2.5 * mad_median;
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
        println!("  Debug - Rate: {:.3}, Variance: {:.3}", rate, variance);
        
        (rate, variance)
    }

    fn estimate_parameters(&self) -> Option<(f64, f64)> {
        let (rate, variance) = self.calculate_moments();
        
        let ratio = variance/(rate * self.window_size/50.0);
        println!("  Debug - Var/Mean Ratio: {:.3}", ratio);
        
        // Optimierte Korrekturen für verschiedene Raten
        let alpha_correction = if rate > 200.0 {
            1.8  // Unverändert für sehr hohe Raten
        } else if rate > 100.0 {
            1.2  // Leicht erhöht für Sells (von 1.1)
        } else {
            1.15  // Leicht erhöht für niedrige Raten
        };

        // Ratio-Korrektur
        let ratio_correction = if ratio < 1.0 {
            ratio.powf(0.7)
        } else {
            ratio.powf(0.3)
        };
        
        let raw_ratio = ratio.min(3.0).max(0.5);
        let alpha_beta_ratio = raw_ratio * alpha_correction * ratio_correction;
        
        println!("  Debug - Korrekturen: rate={:.3}, ratio={:.3}", 
            alpha_correction, ratio_correction);
        
        // Feinere β-Faktoren
        let beta_factor = if rate > 200.0 {
            0.85  // Unverändert für sehr hohe Raten
        } else if rate > 100.0 {
            0.83  // Leicht reduziert für Sells (von 0.85)
        } else {
            0.80  // Leicht erhöht für niedrige Raten
        };
        
        let beta = rate * beta_factor;
        let alpha = alpha_beta_ratio * beta;
        
        println!("  Debug - α/β={:.3}, α={:.3}, β={:.3}", 
            alpha_beta_ratio, alpha, beta);
        
        Some((alpha, beta))
    }
}

// Simuliere getrennte Buy/Sell Prozesse
fn simulate_hawkes_pair(
    lambda_0_buy: f64, 
    alpha_buy: f64, 
    beta_buy: f64,
    lambda_0_sell: f64,
    alpha_sell: f64,
    beta_sell: f64,
    duration: f64
) -> (Vec<i64>, Vec<i64>) {
    let mut rng = rand::thread_rng();
    let exp = Exp::new(1.0).unwrap();
    
    let mut buy_times = Vec::new();
    let mut sell_times = Vec::new();
    let mut t_buy = 0.0;
    let mut t_sell = 0.0;
    let mut lambda_buy = lambda_0_buy;
    let mut lambda_sell = lambda_0_sell;
    
    // Separate Schleifen für Buy und Sell
    while t_buy < duration {
        let wait_buy = exp.sample(&mut rng) / lambda_buy;
        t_buy += wait_buy;
        if t_buy < duration {
            buy_times.push((t_buy * 1000.0) as i64);
            lambda_buy = lambda_0_buy + alpha_buy * buy_times.iter()
                .map(|&s| (-beta_buy * (t_buy - s as f64 / 1000.0)).exp())
                .sum::<f64>();
        }
    }
    
    while t_sell < duration {
        let wait_sell = exp.sample(&mut rng) / lambda_sell;
        t_sell += wait_sell;
        if t_sell < duration {
            sell_times.push((t_sell * 1000.0) as i64);
            lambda_sell = lambda_0_sell + alpha_sell * sell_times.iter()
                .map(|&s| (-beta_sell * (t_sell - s as f64 / 1000.0)).exp())
                .sum::<f64>();
        }
    }
    
    // Debug-Ausgaben
    println!("Simuliert: {} Buys, {} Sells", buy_times.len(), sell_times.len());
    
    (buy_times, sell_times)
}

#[derive(Debug, Deserialize)]
struct Trade {
    timestamp: i64,
    side: String,  // "Buy" oder "Sell"
}

fn main() {
    let duration = 300.0;  // 5 Minuten Fenster
    
    // Historische Trades laden (z.B. aus CSV)
    let trades: Vec<Trade> = load_historical_trades("btc_trades.csv");
    
    // Nach Buy/Sell trennen
    let (buy_times, sell_times): (Vec<_>, Vec<_>) = trades
        .into_iter()
        .partition(|t| t.side == "Buy");
    
    // Buy-Estimator
    let mut buy_estimator = HawkesEstimator::new(duration);
    for trade in buy_times {
        buy_estimator.add_timestamp(trade.timestamp);
    }
    
    // Sell-Estimator
    let mut sell_estimator = HawkesEstimator::new(duration);
    for trade in sell_times {
        sell_estimator.add_timestamp(trade.timestamp);
    }
    
    // Parameter schätzen...
} 