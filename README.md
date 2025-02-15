# Hawkes Process for High-Frequency Trading Analysis

This project implements a real-time Hawkes process analysis for high-frequency trading data from Bybit's BTC-USDT market. It captures trade events, estimates Hawkes parameters, and provides visualization tools for analyzing market microstructure.

## Project Structure

```
hawkes_live/
├── src/
│   ├── main.rs         # Main Rust implementation for data collection
│   └── hawkes_model.rs # Hawkes process parameter estimation
├── validate_hawkes_estimates.py # Python visualization and validation
└── Cargo.toml          # Rust dependencies
```

## Features

- Real-time trade data collection from Bybit WebSocket
- Hawkes process parameter estimation (α, β)
- Separate analysis for buy and sell orders
- Statistical validation and visualization
- Trade intensity estimation
- Clustering effect analysis

## Requirements

### Rust Dependencies
- tokio (async runtime)
- tokio-tungstenite (WebSocket)
- serde (serialization)
- nalgebra (matrix operations)
- chrono (time handling)

### Python Dependencies
- pandas
- numpy
- matplotlib
- datetime

## Output Analysis

The program provides:

1. Trade Volume Analysis
   - Buy/Sell volume per minute
   - Trade count statistics

2. Hawkes Parameters
   - α (alpha): excitation magnitude
   - β (beta): decay rate
   - α/β ratio: clustering effect measure

3. Visualizations
   - Trade volume over time
   - Parameter evolution
   - Intensity comparison
   - Clustering effect analysis

## License

MIT License 