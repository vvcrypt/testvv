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

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd hawkes_live
```

2. Install Rust dependencies:
```bash
cargo build
```

3. Install Python dependencies:
```bash
pip install pandas numpy matplotlib
```

## Usage

1. Start data collection:
```bash
cargo run
```
This will connect to Bybit's WebSocket and start collecting trade data.

2. After collecting sufficient data (recommended: 5+ minutes), run the analysis:
```bash
python3 validate_hawkes_estimates.py
```

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

## Parameter Interpretation

- α/β ratio < 0.8: System stability
- Higher α: Stronger market reaction to trades
- Higher β: Faster decay of trade impact
- Higher clustering: More concentrated trading activity

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your chosen license] 