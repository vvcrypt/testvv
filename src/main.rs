use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use futures::{SinkExt, StreamExt};
use serde_json::json;
use url::Url;
use serde::Deserialize;
use ninjabook::event::Event;
use ninjabook::orderbook::Orderbook;
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use rand::prelude::*;
use rand_distr::{Distribution, Exp};
use std::fs::File;
use std::io::Write;
mod hawkes_model;
use hawkes_model::HawkesModel;

#[derive(Debug, Deserialize, Clone)]
struct BybitTrade {
    T: i64,        // Timestamp
    s: String,     // Symbol
    S: String,     // Side (Buy/Sell)
    v: String,     // Volume
    p: String,     // Price
    i: String,     // Trade ID
}

#[derive(Debug, Deserialize)]
struct TradeMessage {
    topic: String,
    #[serde(rename = "type")]
    msg_type: String,
    data: Vec<BybitTrade>
}

struct HawkesData {
    timestamp: i64,  // Millisekunden-Timestamp
    is_buy: bool,
}

struct MinuteData {
    timestamp: i64,  // Minuten-Timestamp
    trades: Vec<HawkesData>,
}

impl MinuteData {
    fn new(timestamp: i64) -> Self {
        Self {
            timestamp: (timestamp / 60) * 60,
            trades: Vec::new(),
        }
    }

    fn add_trade(&mut self, trade: &BybitTrade) {
        let hawkes_trade = HawkesData {
            timestamp: trade.T,
            is_buy: trade.S == "Buy",
        };
        self.trades.push(hawkes_trade);
    }

    fn print_stats(&self) {
        let (buys, sells): (Vec<_>, Vec<_>) = self.trades.iter()
            .partition(|t| t.is_buy);
        
        println!("\nMinute {} Statistics:", self.timestamp);
        println!("Buy Trades: {}, Sell Trades: {}", buys.len(), sells.len());
    }
}

struct TradeProcessor {
    current_minute: MinuteData,
    history: VecDeque<MinuteData>,
    current_buy_count: usize,
    current_sell_count: usize,
    buy_model: HawkesModel,
    sell_model: HawkesModel,
    log_file: File,
}

impl TradeProcessor {
    fn new() -> Self {
        let mut log_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("hawkes_results.csv")
            .expect("Failed to create log file");
        
        writeln!(log_file, "timestamp,side,alpha,beta,buy_count,sell_count,ratio")
            .expect("Failed to write header");
        
        log_file.flush().expect("Failed to flush buffer");

        Self {
            current_minute: MinuteData::new(chrono::Utc::now().timestamp()),
            history: VecDeque::with_capacity(5),
            current_buy_count: 0,
            current_sell_count: 0,
            buy_model: HawkesModel::new(60.0),
            sell_model: HawkesModel::new(60.0),
            log_file,
        }
    }

    fn process_trade(&mut self, trade: BybitTrade) {
        // Check if new minute begins
        if trade.T / 1000 / 60 != self.current_minute.timestamp / 60 {
            println!("\n=== Minute {} ===", self.current_minute.timestamp);
            println!("Trades: {} Buys, {} Sells", 
                self.current_buy_count,
                self.current_sell_count
            );
            
            // Hawkes parameters for last minute
            if let Some((alpha, beta)) = self.buy_model.estimate_parameters() {
                println!("Buy  Hawkes: α={:.3}, β={:.3}, α/β={:.3}", 
                    alpha, beta, alpha/beta);
            }
            if let Some((alpha, beta)) = self.sell_model.estimate_parameters() {
                println!("Sell Hawkes: α={:.3}, β={:.3}, α/β={:.3}", 
                    alpha, beta, alpha/beta);
            }
            println!("-------------------");
            
            // Process minute transition
            let old_minute = std::mem::replace(
                &mut self.current_minute, 
                MinuteData::new(trade.T / 1000)
            );
            
            self.history.push_back(old_minute);
            if self.history.len() > 5 {
                self.history.pop_front();
            }

            self.current_buy_count = 0;
            self.current_sell_count = 0;
        }

        // Process trade
        match trade.S.as_str() {
            "Buy" => {
                self.current_buy_count += 1;
                self.current_minute.add_trade(&trade);
                self.buy_model.add_timestamp(trade.T);
            },
            "Sell" => {
                self.current_sell_count += 1;
                self.current_minute.add_trade(&trade);
                self.sell_model.add_timestamp(trade.T);
            },
            _ => {}
        }

        // Write parameters to file
        if let Some((alpha, beta)) = if trade.S == "Buy" {
            self.buy_model.estimate_parameters()
        } else {
            self.sell_model.estimate_parameters()
        } {
            let side = if trade.S == "Buy" { "Buy" } else { "Sell" };
            let ratio = self.current_buy_count as f64 / self.current_sell_count.max(1) as f64;
            
            writeln!(
                self.log_file,
                "{},{},{:.3},{:.3},{},{},{:.3}",
                trade.T,
                side,
                alpha,
                beta,
                self.current_buy_count,
                self.current_sell_count,
                ratio
            ).expect("Failed to write data");
            
            self.log_file.flush().expect("Failed to flush buffer");
        }
    }
}

pub struct HawkesEstimator {
    timestamps: VecDeque<i64>,  // Ring-Buffer statt Vec
    window_size: f64,
    capacity: usize,            // Maximale Anzahl Events
}

impl HawkesEstimator {
    pub fn new(window_size: f64) -> Self {
        let capacity = (window_size * 1000.0) as usize; // Geschätzte Kapazität
        Self {
            timestamps: VecDeque::with_capacity(capacity),
            window_size,
            capacity,
        }
    }

    pub fn add_timestamp(&mut self, ts: i64) {
        // Alte Events entfernen (älter als window_size)
        let cutoff = ts - (self.window_size * 1000.0) as i64;
        while let Some(&oldest) = self.timestamps.front() {
            if oldest < cutoff {
                self.timestamps.pop_front();
            } else {
                break;
            }
        }

        // Neues Event hinzufügen
        self.timestamps.push_back(ts);
    }

    fn estimate_beta(events: &[i64], lag_tau: f64) -> f64 {
        let max_lag = 10;  // Anzahl der Lags für ACF
        let mut acf = Vec::new();
        
        for lag in 1..=max_lag {
            let lag_time = lag as f64 * lag_tau / max_lag as f64;
            let mut sum = 0.0;
            let mut count = 0;
            
            for (i, &t1) in events.iter().enumerate() {
                for &t2 in &events[i+1..] {
                    let dt = (t2 - t1) as f64 / 1000.0;
                    if dt <= lag_time {
                        sum += dt;
                        count += 1;
                    }
                }
            }
            
            if count > 0 {
                acf.push((lag_time, sum / count as f64));
            }
        }
        
        // Fit exponential decay to ACF
        if acf.len() >= 2 {
            let (_, y1) = acf[0];
            let (t2, y2) = acf[1];
            -1.0 * (y2/y1).ln() / t2
        } else {
            1.0 / lag_tau  // Fallback
        }
    }

    pub fn estimate_parameters(&self) -> Option<(f64, f64)> {
        if self.timestamps.len() < 10 { return None; }
        
        let minute = 60.0;  // 1 Minute in Sekunden
        let delta_t = 1.0;  // 1s Window innerhalb der Minute
        let lag_tau = minute;  // Lag = 1 Minute
        
        // Nur Events der letzten Minute verwenden
        let current_time = *self.timestamps.back()? as f64 / 1000.0;
        let minute_start = (current_time / minute).floor() * minute;
        
        let minute_events: Vec<_> = self.timestamps.iter()
            .copied()
            .filter(|&ts| {
                let ts_sec = ts as f64 / 1000.0;
                ts_sec >= minute_start && ts_sec < minute_start + minute
            })
            .collect();
            
        if minute_events.is_empty() { return None; }
        
        // 1. N(t) und L(t) für jedes Window in der Minute
        let mut windows = Vec::new();
        let mut t = minute_start;
        
        while t < minute_start + minute {
            let window_events = minute_events.iter()
                .filter(|&&ts| {
                    let ts_sec = ts as f64 / 1000.0;
                    ts_sec >= t && ts_sec < t + delta_t
                })
                .count();
                
            let lag_events = minute_events.iter()
                .filter(|&&ts| {
                    let ts_sec = ts as f64 / 1000.0;
                    ts_sec >= t - lag_tau && ts_sec < t
                })
                .count();
                
            windows.push((window_events as f64, lag_events as f64));
            t += delta_t;
        }
        
        // 2. Parameter-Optimierung wie im Paper
        let (sum_n, sum_l, sum_nl, sum_l2) = windows.iter()
            .fold((0.0, 0.0, 0.0, 0.0), |acc, &(n, l)| {
                (acc.0 + n, acc.1 + l, acc.2 + n*l, acc.3 + l*l)
            });
            
        let n = windows.len() as f64;
        let alpha = (n * sum_nl - sum_n * sum_l) / (n * sum_l2 - sum_l * sum_l);
        let lambda_inf = (sum_n - alpha * sum_l) / (n * delta_t);
        
        // 3. β aus ACF schätzen
        let beta = Self::estimate_beta(&minute_events, lag_tau);
        
        // Validierung
        if alpha > 0.0 && alpha < beta && alpha/beta < 0.8 {
            Some((alpha, beta))
        } else {
            None
        }
    }

    pub fn diagnostics(&self) -> Option<DiagnosticStats> {
        if let Some((alpha, beta)) = self.estimate_parameters() {
            let events_per_window = self.timestamps.len() as f64 / self.window_size;
            let branching_ratio = alpha / beta;
            
            println!("\nDiagnostik:");
            println!("Events/Sekunde: {:.1}", events_per_window);
            println!("Branching Ratio: {:.3}", branching_ratio);
            println!("Stabilität: {}", if branching_ratio < 1.0 { "Stabil" } else { "Instabil" });
            
            Some(DiagnosticStats {
                events_per_window,
                branching_ratio,
                alpha,
                beta
            })
        } else {
            None
        }
    }
}

pub struct ThreadSafeHawkesEstimator {
    inner: Arc<Mutex<HawkesEstimator>>,
}

impl ThreadSafeHawkesEstimator {
    pub fn new(window_size: f64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HawkesEstimator::new(window_size))),
        }
    }

    pub fn add_timestamp(&self, ts: i64) {
        let mut estimator = self.inner.lock().unwrap();
        estimator.add_timestamp(ts);
    }

    pub fn estimate_parameters(&self) -> Option<(f64, f64)> {
        let estimator = self.inner.lock().unwrap();
        estimator.estimate_parameters()
    }
}

pub struct BatchHawkesEstimator {
    estimators: HashMap<String, ThreadSafeHawkesEstimator>,
    window_size: f64,
}

impl BatchHawkesEstimator {
    pub fn new(window_size: f64) -> Self {
        Self {
            estimators: HashMap::new(),
            window_size,
        }
    }

    pub fn process_update(&mut self, instrument: &str, ts: i64) {
        self.estimators
            .entry(instrument.to_string())
            .or_insert_with(|| ThreadSafeHawkesEstimator::new(self.window_size))
            .add_timestamp(ts);
    }

    pub fn get_parameters(&self, instrument: &str) -> Option<(f64, f64)> {
        self.estimators
            .get(instrument)
            .and_then(|est| est.estimate_parameters())
    }
}

async fn process_messages() -> Result<(), Box<dyn std::error::Error>> {
    let url = Url::parse("wss://stream.bybit.com/v5/public/linear")?;
    let (ws_stream, _) = connect_async(url).await?;
    let (mut write, mut read) = ws_stream.split();
    
    println!("WebSocket verbunden!");

    let subscribe_msg = json!({
        "op": "subscribe",
        "args": ["publicTrade.BTCUSDT"]
    });
    write.send(Message::Text(subscribe_msg.to_string())).await?;
    println!("Subscribe-Nachricht gesendet");
    
    let mut processor = TradeProcessor::new();
    
    while let Some(msg) = read.next().await {
        if let Ok(Message::Text(text)) = msg {
            if let Ok(trade_msg) = serde_json::from_str::<TradeMessage>(&text) {
                for trade in trade_msg.data {
                    processor.process_trade(trade);
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
pub struct DiagnosticStats {
    events_per_window: f64,
    branching_ratio: f64,
    alpha: f64,
    beta: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Live-Trading Start ===");
    process_messages().await  // Dann starte das Live-Trading
}

fn test_known_parameters() {
    let test_cases = vec![
        // (α, β, λ₀, erwartete Events/Sekunde)
        (0.3, 2.0, 5.0,  7.1),   // Niedrige Intensität
        (0.5, 3.0, 10.0, 14.3),  // Mittlere Intensität
        (0.7, 4.0, 15.0, 21.4),  // Hohe Intensität
    ];
    
    for (true_alpha, true_beta, lambda_0, expected_rate) in test_cases {
        println!("\nTest mit α={:.1}, β={:.1}, λ₀={:.1} (Erwartete Rate: {:.1})", 
            true_alpha, true_beta, lambda_0, expected_rate);
        
        let n_tests = 5;
        for i in 0..n_tests {
            let simulated_times = simulate_hawkes(lambda_0, true_alpha, true_beta, 300.0);
            let actual_rate = simulated_times.len() as f64 / 300.0;
            
            let mut estimator = HawkesEstimator::new(300.0);
            for &t in &simulated_times {
                estimator.add_timestamp(t);
            }
            
            if let Some((est_alpha, est_beta)) = estimator.estimate_parameters() {
                println!("  Run {}: α={:.2}, β={:.2}, Rate={:.1} (Fehler: α={:+.1}%, β={:+.1}%)", 
                    i+1,
                    est_alpha,
                    est_beta,
                    actual_rate,
                    100.0 * (est_alpha - true_alpha)/true_alpha,
                    100.0 * (est_beta - true_beta)/true_beta
                );
            }
        }
    }
}

fn simulate_hawkes(lambda_0: f64, alpha: f64, beta: f64, duration: f64) -> Vec<i64> {
    let mut rng = rand::thread_rng();
    let exp = Exp::new(1.0).unwrap();
    
    let mut times = Vec::new();
    let mut t = 0.0;
    let mut lambda = lambda_0;
    
    // Kleinere Zeitschritte für genauere Simulation
    let dt = 0.001;  // 1ms Schritte
    let steps = (duration / dt) as usize;
    
    for _ in 0..steps {
        // Thinning-Algorithmus
        let u = rng.gen::<f64>();
        if u < lambda * dt {
            times.push((t * 1000.0) as i64);
            
            // Update Intensität nur bei Events
            lambda = lambda_0;
            for &past_time in times.iter().rev() {
                let dt = t - (past_time as f64 / 1000.0);
                if dt > 5.0 { break; }
                lambda += alpha * (-beta * dt).exp();
            }
        }
        t += dt;
    }
    
    times
}

fn validate_estimator() {
    // Realistische Parameter für Krypto-Trading
    let true_params = vec![
        // (α, β, λ₀, Beschreibung)
        (2.0, 3.5, 10.0, "Normaler Markt"),
        (3.0, 4.5, 15.0, "Volatiler Markt"),
        (1.5, 3.0, 5.0,  "Ruhiger Markt"),
    ];
    
    for (true_alpha, true_beta, lambda_0, desc) in true_params {
        println!("\nTest: {}", desc);
        
        // Simuliere 10 Minuten Daten
        let times = simulate_hawkes(lambda_0, true_alpha, true_beta, 600.0);
        
        let mut estimator = HawkesEstimator::new(300.0);
        for &t in &times {
            estimator.add_timestamp(t);
        }
        
        if let Some((est_alpha, est_beta)) = estimator.estimate_parameters() {
            println!("Wahr:     α={:.2}, β={:.2}, α/β={:.2}", 
                true_alpha, true_beta, true_alpha/true_beta);
            println!("Geschätzt: α={:.2}, β={:.2}, α/β={:.2}", 
                est_alpha, est_beta, est_alpha/est_beta);
        }
    }
}