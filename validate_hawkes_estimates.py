import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better performance
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

def calculate_true_intensity(trades_df, window='30s'):
    """Calculate the actual trading intensity from trade data"""
    trades_df = trades_df.copy()
    trades_df.set_index('timestamp', inplace=True)
    
    t_grid = pd.date_range(start=trades_df.index.min(),
                          end=trades_df.index.max(),
                          freq='1s')
    
    counts = trades_df.resample(window).size()
    intensity = counts / pd.Timedelta(window).total_seconds()
    intensity = intensity.reindex(t_grid).interpolate()
    
    return intensity

def calculate_hawkes_intensity(trades_df, alpha, beta):
    """Calculate estimated Hawkes intensity λ(t) = λ∞ + Σ(α * exp(-β(t-ti)))"""
    trades_df = trades_df.copy()
    trades_df.set_index('timestamp', inplace=True)
    
    t_grid = pd.date_range(start=trades_df.index.min(),
                          end=trades_df.index.max(),
                          freq='1s')
    
    T = (trades_df.index.max() - trades_df.index.min()).total_seconds()
    N = len(trades_df)
    lambda_inf = N/T * (1 - alpha/beta)
    
    intensity = np.zeros(len(t_grid))
    for t_idx, t in enumerate(t_grid):
        past_events = trades_df.index[trades_df.index < t]
        if len(past_events) > 0:
            dt = (t - past_events).total_seconds()
            intensity[t_idx] = lambda_inf + np.sum(alpha * np.exp(-beta * dt))
        else:
            intensity[t_idx] = lambda_inf
            
    return pd.Series(intensity, index=t_grid)

def plot_results():
    # Load data
    df = pd.read_csv('/Users/home/Downloads/Trading/hawkes_rust/hawkes_live/src/hawkes_results.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Aggregate data per minute
    df['minute'] = df['timestamp'].dt.floor('min')
    minute_stats = df.groupby(['minute', 'side']).agg({
        'alpha': 'mean',
        'beta': 'mean',
        'timestamp': 'count'
    }).reset_index()
    
    # Create plots
    plt.figure(figsize=(15, 20))
    
    # Plot 1: Trading volume
    plt.subplot(4, 1, 1)
    for side in ['Buy', 'Sell']:
        side_data = minute_stats[minute_stats['side'] == side]
        plt.plot(side_data['minute'], side_data['timestamp'], 
                 label=f'{side} Volume', alpha=0.7)

    plt.title('Trading Volume per Minute')
    plt.ylabel('Number of Trades')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Hawkes parameter (α)
    plt.subplot(4, 1, 2)
    for side in ['Buy', 'Sell']:
        side_data = minute_stats[minute_stats['side'] == side]
        plt.plot(side_data['minute'], side_data['alpha'], 
                 label=f'α ({side})', alpha=0.7)

    plt.title('Alpha Parameter over Time')
    plt.ylabel('α Value')
    plt.legend()
    plt.grid(True)

    # Plot 3: Hawkes parameter (β)
    plt.subplot(4, 1, 3)
    for side in ['Buy', 'Sell']:
        side_data = minute_stats[minute_stats['side'] == side]
        plt.plot(side_data['minute'], side_data['beta'], 
                 label=f'β ({side})', alpha=0.7)

    plt.title('Beta Parameter over Time')
    plt.ylabel('β Value')
    plt.legend()
    plt.grid(True)

    # Plot 4: α/β Ratio
    plt.subplot(4, 1, 4)
    for side in ['Buy', 'Sell']:
        side_data = minute_stats[minute_stats['side'] == side]
        ratio = side_data['alpha'] / side_data['beta']
        plt.plot(side_data['minute'], ratio, 
                 label=f'α/β Ratio ({side})', alpha=0.7)

    plt.axhline(y=0.8, color='r', linestyle='--', label='Stability Threshold')
    plt.title('Clustering Effect (α/β Ratio)')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # Statistical analysis
    print("\nStatistical Analysis (aggregated per minute):")
    for side in ['Buy', 'Sell']:
        side_data = minute_stats[minute_stats['side'] == side]
        
        print(f"\n{side} Trades:")
        print(f"Average α/β Ratio: {(side_data['alpha']/side_data['beta']).mean():.3f}")
        print(f"Average α: {side_data['alpha'].mean():.3f}")
        print(f"Average β: {side_data['beta'].mean():.3f}")
        print(f"Average trades per minute: {side_data['timestamp'].mean():.1f}")
        print(f"Max trades per minute: {side_data['timestamp'].max():.0f}")
        print(f"Max α: {side_data['alpha'].max():.3f}")
        print(f"Max β: {side_data['beta'].max():.3f}")
        print(f"α-β correlation: {side_data['alpha'].corr(side_data['beta']):.3f}")

if __name__ == "__main__":
    plot_results()
    plt.show() 