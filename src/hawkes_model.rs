use std::collections::VecDeque;
use nalgebra as na;

// ODEs System für die Momente
#[derive(Debug)]
struct ODESystem {
    a: na::Matrix5<f64>,
    b: na::Vector5<f64>,
}

impl ODESystem {
    fn new(lambda_inf: f64, alpha: f64, beta: f64) -> Self {
        let mut a = na::Matrix5::zeros();
        let mut b = na::Vector5::zeros();
        
        // Matrix A wie im Paper definiert
        a[(0, 0)] = alpha - beta;
        a[(1, 0)] = 1.0;
        a[(2, 0)] = alpha.powi(2) + 2.0 * beta * lambda_inf;
        a[(2, 2)] = 2.0 * (alpha - beta);
        a[(3, 0)] = alpha;
        a[(3, 1)] = beta * lambda_inf;
        a[(3, 2)] = 1.0;
        a[(3, 3)] = alpha - beta;
        a[(4, 0)] = 1.0;
        a[(4, 3)] = 2.0;

        // Vektor B
        b[0] = beta * lambda_inf;
        
        ODESystem { a, b }
    }

    // Approximierte Matrix-Exponential-Funktion
    fn matrix_exp(&self, t: f64) -> na::Matrix5<f64> {
        let mut result = na::Matrix5::identity();
        let mut term = na::Matrix5::identity();
        let mut factorial = 1.0;
        
        for i in 1..10 {  // 10 Terme für die Reihenentwicklung
            term = term * &self.a * (t / factorial);
            result += &term;
            factorial *= (i + 1) as f64;
        }
        
        result
    }

    fn solve(&self, t: f64, z0: &na::Vector5<f64>) -> na::Vector5<f64> {
        // Matrix-Exponential und Integration für ODE-Lösung
        let exp_at = self.matrix_exp(t);
        let integral = (0..100).map(|i| {
            let ti = t * (i as f64) / 100.0;
            exp_at * self.matrix_exp(-ti) * &self.b
        }).sum::<na::Vector5<f64>>() * (t / 100.0);
        
        exp_at * z0 + integral
    }
}

pub struct HawkesModel {
    timestamps: VecDeque<i64>,
    window_size: f64,
    dt: f64,          
    tau: f64,         
    delta: f64,
    ode_system: ODESystem,
}

impl HawkesModel {
    pub fn new(window_size: f64) -> Self {
        Self {
            timestamps: VecDeque::new(),
            window_size,
            dt: 0.1,          // Feinere Zeitauflösung
            tau: 300.0,       // Längeres Schätzfenster
            delta: 1.0,
            ode_system: ODESystem::new(1.0, 0.5, 1.0), // Standardwerte
        }
    }

    fn calculate_intensity(&self, t: f64, _lambda_0: f64, lambda_inf: f64, alpha: f64, beta: f64) -> f64 {
        if alpha >= beta {
            return lambda_inf;  // Fallback für instabile Fälle
        }
        
        let base = lambda_inf;  // Grundintensität
        
        // Nur relevante vergangene Events berücksichtigen
        let cutoff = t - 5.0/beta;  // ~5 Halbwertszeiten
        
        let excitation: f64 = self.timestamps.iter()
            .rev()  // Rückwärts iterieren für Effizienz
            .take_while(|&&ts| {
                let s = ts as f64 / 1000.0;
                s >= cutoff && s < t
            })
            .map(|&ts| {
                let s = ts as f64 / 1000.0;
                alpha * (-beta * (t - s)).exp()
            })
            .sum();
            
        base + excitation
    }

    // Levenberg-Marquardt Implementierung
    fn levenberg_marquardt(
        &self,
        initial_params: (f64, f64, f64),
        empirical_moments: &[f64],
        max_iter: usize,
    ) -> Option<(f64, f64, f64)> {
        let mut params = na::Vector3::new(
            initial_params.0,
            initial_params.1,
            initial_params.2,
        );
        
        let mut lambda = 0.001;  // Dämpfungsfaktor
        let mut prev_error = f64::INFINITY;
        
        for _ in 0..max_iter {
            // Jacobi-Matrix berechnen
            let j = self.compute_jacobian(&params);
            
            // Residuen berechnen
            let r = self.compute_residuals(&params, empirical_moments);
            
            // Normal-Gleichungen
            let mut normal = j.transpose() * &j;
            
            // Dämpfung hinzufügen
            for i in 0..3 {
                normal[(i, i)] += lambda;
            }
            
            // Update-Richtung berechnen
            let delta = match normal.try_inverse() {
                Some(inv) => -inv * j.transpose() * r,
                None => return None,
            };
            
            // Neue Parameter testen
            let new_params = params + delta;
            
            // Prüfen ob neue Parameter besser sind
            let new_error = self.compute_error(&new_params, empirical_moments);
            
            if new_error < prev_error {
                // Update akzeptieren
                params = new_params;
                prev_error = new_error;
                lambda /= 10.0;
            } else {
                // Update ablehnen
                lambda *= 10.0;
            }
            
            // Konvergenzcheck
            if delta.norm() < 1e-6 {
                return Some((params[0], params[1], params[2]));
            }
        }
        
        None
    }

    fn compute_jacobian(&self, params: &na::Vector3<f64>) -> na::Matrix3<f64> {
        let eps = 1e-6;
        let mut j = na::Matrix3::zeros();
        
        // Numerische Differentiation für jeden Parameter
        for i in 0..3 {
            let mut params_plus = params.clone();
            params_plus[i] += eps;
            
            let mut params_minus = params.clone();
            params_minus[i] -= eps;
            
            let derivative = (self.theoretical_moments(&params_plus) -
                            self.theoretical_moments(&params_minus)) / (2.0 * eps);
            
            for k in 0..3 {
                j[(k, i)] = derivative[k];
            }
        }
        
        j
    }

    fn compute_residuals(
        &self,
        params: &na::Vector3<f64>,
        empirical: &[f64],
    ) -> na::Vector3<f64> {
        let theoretical = self.theoretical_moments(params);
        na::Vector3::from_iterator(
            empirical.iter()
                    .zip(theoretical.iter())
                    .map(|(e, t)| e - t)
        )
    }

    fn compute_error(
        &self,
        params: &na::Vector3<f64>,
        empirical: &[f64],
    ) -> f64 {
        let r = self.compute_residuals(params, empirical);
        r.dot(&r)
    }

    fn theoretical_moments(&self, params: &na::Vector3<f64>) -> na::Vector3<f64> {
        let lambda_inf = params[0];
        let alpha = params[1];
        let beta = params[2];
        
        if alpha >= beta {
            return na::Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        }
        
        let lambda = lambda_inf / (1.0 - alpha / beta);
        let kappa = 1.0 / (1.0 - alpha / beta);
        let gamma = beta - alpha;
        
        // Erwartungswert
        let mean = lambda * self.tau;
        
        // Varianz
        let var = lambda * self.tau * (
            kappa.powi(2) +
            (1.0 - kappa.powi(2)) * (1.0 - (-gamma * self.tau).exp()) / gamma
        );
        
        // Autokovarianz bei lag τ
        let acov = lambda_inf * beta * alpha * (2.0 * beta - alpha) *
            ((alpha - beta) * self.tau).exp().powi(2) /
            (2.0 * (alpha - beta).powi(4)) *
            ((alpha - beta) * self.tau).exp();
        
        na::Vector3::new(mean, var, acov)
    }

    fn compute_empirical_moments(&self) -> Vec<f64> {
        let current_time = *self.timestamps.back().unwrap() as f64 / 1000.0;
        let window_start = current_time - self.tau;
        
        // Events im Fenster
        let window_events: Vec<_> = self.timestamps.iter()
            .copied()
            .filter(|&ts| {
                let ts_sec = ts as f64 / 1000.0;
                ts_sec >= window_start && ts_sec < current_time
            })
            .collect();
            
        let n = window_events.len() as f64;
        
        // Mittelwert
        let mean = n / self.tau;
        
        // Varianz (durch Binning)
        let n_bins = 30;
        let bin_size = self.tau / n_bins as f64;
        let mut counts = vec![0; n_bins];
        
        for &ts in &window_events {
            let ts_sec = ts as f64 / 1000.0;
            let bin = ((ts_sec - window_start) / bin_size) as usize;
            if bin < n_bins {
                counts[bin] += 1;
            }
        }
        
        let mean_count = n / n_bins as f64;
        let variance = counts.iter()
            .map(|&c| (c as f64 - mean_count).powi(2))
            .sum::<f64>() / (n_bins - 1) as f64;
            
        // Autokovarianz
        let mut acov = 0.0;
        let mut count = 0;
        
        for (i, &t1) in window_events.iter().enumerate() {
            for &t2 in &window_events[i+1..] {
                let dt = (t2 - t1) as f64 / 1000.0;
                if dt <= self.tau {
                    acov += dt;
                    count += 1;
                }
            }
        }
        
        let acov = if count > 0 { acov / count as f64 } else { 0.0 };
        
        vec![mean, variance, acov]
    }

    fn fast_initial_estimate(&self) -> Option<(f64, f64)> {
        if self.timestamps.len() < 10 { return None; }
        
        let current_time = *self.timestamps.back()? as f64 / 1000.0;
        let window_start = current_time - self.tau;
        
        // Events im Fenster
        let window_events: Vec<_> = self.timestamps.iter()
            .copied()
            .filter(|&ts| {
                let ts_sec = ts as f64 / 1000.0;
                ts_sec >= window_start && ts_sec < current_time
            })
            .collect();
            
        if window_events.is_empty() { return None; }
        
        // Beta aus mittlerem Inter-Event-Abstand schätzen
        let mut inter_times: Vec<f64> = window_events.windows(2)
            .map(|w| (w[1] - w[0]) as f64 / 1000.0)
            .collect();
            
        let beta = if !inter_times.is_empty() {
            1.0 / inter_times.iter().sum::<f64>() * inter_times.len() as f64
        } else {
            2.0 // Fallback
        };
        
        // Alpha aus Event-Rate schätzen
        let rate = window_events.len() as f64 / self.tau;
        let alpha = beta * (rate / (rate + beta));
        
        if alpha > 0.0 && alpha < beta {
            Some((alpha, beta))
        } else {
            None
        }
    }

    pub fn estimate_parameters(&self) -> Option<(f64, f64)> {
        if self.timestamps.len() < 10 { return None; }
        
        // Erste Schätzung mit der schnellen Methode
        let (alpha_init, beta_init) = self.fast_initial_estimate()?;
        
        // Empirische Momente berechnen
        let empirical_moments = self.compute_empirical_moments();
        
        // Levenberg-Marquardt mit der ersten Schätzung als Startwert
        if let Some((lambda_inf, alpha, beta)) = self.levenberg_marquardt(
            (1.0, alpha_init, beta_init),
            &empirical_moments,
            100
        ) {
            if self.validate_parameters(alpha, beta) {
                return Some((alpha, beta));
            }
        }
        
        // Fallback auf die schnelle Schätzung
        Some((alpha_init, beta_init))
    }

    fn validate_parameters(&self, alpha: f64, beta: f64) -> bool {
        let ratio = alpha / beta;
        
        // Weniger strikte Kriterien
        let valid_ratio = 0.05 < ratio && ratio < 0.8;  // Erweiterte Grenzen
        let valid_beta = 0.5 < beta && beta < 10.0;     // Erweiterte Grenzen
        let valid_intensity = {
            let lambda_inf = beta * (1.0 - ratio);
            0.1 < lambda_inf && lambda_inf < 50.0       // Erweiterte Grenzen
        };
        
        valid_ratio && valid_beta && valid_intensity
    }

    pub fn add_timestamp(&mut self, ts: i64) {
        let cutoff = ts - (self.window_size * 1000.0) as i64;
        self.timestamps.retain(|&t| t >= cutoff);
        self.timestamps.push_back(ts);
    }
} 