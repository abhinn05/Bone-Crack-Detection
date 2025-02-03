import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def load_antenna_files(directory='.'):
    """
    Load all antenna S2P files from the specified directory
    Returns a list of NetworkZ objects sorted by antenna number
    """
    # Find all s2p files matching the pattern antenna*.s2p
    s2p_files = glob.glob(str(Path(directory) / "antenna*.s2p"))

    # Sort files by antenna number
    s2p_files.sort(key=lambda x: int(''.join(filter(str.isdigit, Path(x).stem))))

    # Load networks
    networks = []
    for file in s2p_files:
        try:
            network = rf.Network(file)
            networks.append(network)
            print(f"Loaded {Path(file).name}")
        except Exception as e:
            print(f"Error loading {Path(file).name}: {e}")

    return networks

def plot_s_parameters_multi(networks, fig_size=(15, 10)):
    """
    Create visualization of S-parameters for multiple antennas
    """
    n_antennas = len(networks)

    # Create figure with subplots for magnitude and phase
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size)

    # Color cycle for different antennas
    colors = plt.cm.rainbow(np.linspace(0, 1, n_antennas))

    # Plot S11 Magnitude for all antennas
    for i, ntwk in enumerate(networks):
        freq_ghz = ntwk.f / 1e9  # Convert to GHz
        s11_mag_db = 20 * np.log10(np.abs(ntwk.s11.s.flatten()))  # Convert to dB
        ax1.plot(freq_ghz, s11_mag_db, color=colors[i],
                label=f'Antenna {i+1}', linewidth=1.5)

    ax1.set_title('S11 Magnitude')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('|S11| (dB)')
    ax1.grid(True)
    ax1.legend()

    # Plot S11 Phase for all antennas
    for i, ntwk in enumerate(networks):
        freq_ghz = ntwk.f / 1e9
        s11_phase = np.angle(ntwk.s11.s.flatten(), deg=True)
        ax2.plot(freq_ghz, s11_phase, color=colors[i],
                label=f'Antenna {i+1}', linewidth=1.5)

    ax2.set_title('S11 Phase')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    return fig

def plot_smith_charts(networks, fig_size=(15, 5)):
    """
    Create Smith chart visualization for multiple antennas
    """
    n_antennas = len(networks)

    # Calculate number of rows and columns for subplots
    n_cols = min(3, n_antennas)  # Maximum 3 columns
    n_rows = (n_antennas - 1) // n_cols + 1

    fig = plt.figure(figsize=(fig_size[0], fig_size[1] * n_rows))

    for i, ntwk in enumerate(networks):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ntwk.plot_s_smith(m=0, n=0, ax=ax, show_legend=False)
        ax.set_title(f'Antenna {i+1} Smith Chart')

    plt.tight_layout()
    return fig

def visualize_all_antennas(directory='.', save_plots=False):
    """
    Create complete visualization for all antenna data
    """
    # Load all antenna files
    networks = load_antenna_files(directory)

    if not networks:
        print("No antenna files found!")
        return

    # Create standard S-parameter plots
    s_param_fig = plot_s_parameters_multi(networks)

    # Create Smith charts
    smith_fig = plot_smith_charts(networks)

    if save_plots:
        s_param_fig.savefig('s_parameters.png', dpi=300, bbox_inches='tight')
        smith_fig.savefig('smith_charts.png', dpi=300, bbox_inches='tight')

    return s_param_fig, smith_fig

def analyze_frequency_response(networks, target_freq=2.4e9):
    """
    Analyze the frequency response around the target frequency
    """
    print(f"\nAnalyzing frequency response around {target_freq/1e9:.1f} GHz:")
    print("-" * 50)

    for i, ntwk in enumerate(networks):
        # Find closest frequency point
        freq_idx = np.abs(ntwk.f - target_freq).argmin()
        freq = ntwk.f[freq_idx]
        s11 = ntwk.s11.s.flatten()[freq_idx]

        # Calculate metrics
        magnitude_db = 20 * np.log10(np.abs(s11))
        phase_deg = np.angle(s11, deg=True)

        print(f"\nAntenna {i+1}:")
        print(f"Frequency: {freq/1e9:.3f} GHz")
        print(f"S11 Magnitude: {magnitude_db:.2f} dB")
        print(f"S11 Phase: {phase_deg:.2f} degrees")

        # Basic assessment
        if magnitude_db < -10:
            print("Good impedance matching")
        elif magnitude_db < -6:
            print("Moderate impedance matching")
        else:
            print("Poor impedance matching")

if __name__ == "__main__":
    # Directory containing the s2p files
    data_dir = '.'

    # Create visualizations
    s_param_fig, smith_fig = visualize_all_antennas(data_dir, save_plots=True)

    # Load networks for analysis
    networks = load_antenna_files(data_dir)

    # Analyze frequency response around 2.4 GHz
    if networks:
        analyze_frequency_response(networks)

    # Show plots
    plt.show()