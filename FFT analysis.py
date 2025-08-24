import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# --- Titre principal ---
st.title("Application d'analyse FFT et comparaison au modèle continu")

# --- Sidebar pour paramètres ---
st.sidebar.title("⚙ Paramètres d'analyse")
start_threshold = st.sidebar.number_input("Exclure les N premières secondes :", min_value=0.0, value=30.0, step=1.0)
end_threshold = st.sidebar.number_input("Exclure les N dernières secondes :", min_value=0.0, value=20.0, step=1.0)
fixed_fundamental = st.sidebar.number_input("Forcer la fréquence fondamentale (Hz, mettre 0 pour auto)", min_value=0.0, value=0.0, step=1.0)
freq_display_max = st.sidebar.number_input("Fréquence max affichée sur la FFT (Hz)", min_value=1.0, value=10.0, step=1.0)
uploaded_file1 = st.sidebar.file_uploader("Chargez le premier fichier CSV", type=["csv"])
uploaded_file2 = st.sidebar.file_uploader("Chargez le deuxième fichier CSV", type=["csv"])

# --- Encadré explicatif ---
st.subheader("💡 Définitions des indicateurs")
st.markdown("""
- **Fréquence fondamentale (Hz)** : fréquence principale du signal correspondant au mouvement moteur.
- **Amplitude fondamentale** : intensité de la fréquence principale.
- **SNR (dB)** : rapport signal/bruit. Plus SNR élevé → signal plus propre.
- **THD (dB)** : distorsion totale des harmoniques. Plus THD faible → signal moins déformé.
- **Bruit (0-10 Hz hors harmoniques)** : perturbations sur le signal.
- **RMSE par rapport au modèle** : écart moyen entre le signal réel et le signal continu idéal.
- **Score global** : combinaison pondérée des indicateurs.
""")

# --- FFT et analyse ---
def analyze_signal(time, signal, fixed_fundamental=0.0):
    dt = time[1] - time[0]
    sig_centered = signal - np.mean(signal)
    fft_vals = np.fft.fft(sig_centered)
    freqs = np.fft.fftfreq(len(sig_centered), d=dt)
    mask = freqs >= 0

    magnitude_pos = np.abs(fft_vals[mask]) / len(sig_centered)
    magnitude_pos[1:] *= 2
    freqs_pos = freqs[mask]

    fundamental_freq = 0
    harmonics = []

    if fixed_fundamental > 0:
        fundamental_freq = fixed_fundamental
        magnitude_interp = np.interp(fundamental_freq, freqs_pos, magnitude_pos)
        harmonics.append((1, fundamental_freq, magnitude_interp))
    elif len(magnitude_pos) > 1:
        idx = np.argmax(magnitude_pos[1:]) + 1
        fundamental_freq = freqs_pos[idx]
        harmonics.append((1, fundamental_freq, magnitude_pos[idx]))

    if fundamental_freq > 0:
        for n in range(2, 11):
            target = n * fundamental_freq
            if target <= freqs_pos[-1]:
                magnitude_interp = np.interp(target, freqs_pos, magnitude_pos)
                harmonics.append((n, target, magnitude_interp))

    tol = 0.01
    noise_power = sum([m**2 for f, m in zip(freqs_pos, magnitude_pos) if 0 <= f <= 10 and all(abs(f-h[1]) > tol for h in harmonics)])

    power_fund = harmonics[0][2]**2 if harmonics else 0
    power_harmo = sum([h[2]**2 for h in harmonics[1:]]) if harmonics else 0

    SNR = 10*np.log10(power_fund / noise_power) if noise_power>0 else np.inf
    THD = 10*np.log10(power_harmo / power_fund) if power_fund>0 else -np.inf
    amp_fundamental = harmonics[0][2] if harmonics else 0

    score_global = 0.4*SNR - 0.3*THD - 0.2*noise_power + 0.1*amp_fundamental

    return freqs_pos, magnitude_pos, fundamental_freq, harmonics, noise_power, SNR, THD, amp_fundamental, score_global

# --- Signal modèle idéal ---
def generate_ideal_signal(time, signal, mode='mean'):
    if mode=='mean':
        return np.full_like(signal, np.mean(signal))
    elif mode=='linear':
        return np.linspace(signal[0], signal[-1], len(signal))
    else:
        raise ValueError("Mode inconnu")

def compare_to_model(time, signal_real, signal_ideal):
    diff = signal_real - signal_ideal
    rmse = np.sqrt(np.mean(diff**2))
    freqs_diff, mag_diff, _, _, noise_power_diff, _, _, _, _ = analyze_signal(time, diff)
    return diff, rmse, freqs_diff, mag_diff, noise_power_diff

# --- Traitement des fichiers ---
if uploaded_file1 and uploaded_file2:
    try:
        df1 = pd.read_csv(uploaded_file1, decimal=',')
        df2 = pd.read_csv(uploaded_file2, decimal=',')
        for i, df in enumerate([df1, df2], start=1):
            if not {'Time','Signal'}.issubset(df.columns):
                st.error(f"Le fichier CSV {i} doit contenir les colonnes 'Time' et 'Signal'")
                st.stop()
        time1, signal1 = df1['Time'].values, df1['Signal'].values
        time2, signal2 = df2['Time'].values, df2['Signal'].values

        # Filtrage temporel
        start_idx1 = np.searchsorted(time1, start_threshold)
        end_idx1 = np.searchsorted(time1, time1[-1]-end_threshold)-1
        time_filtered1, signal_filtered1 = time1[start_idx1:end_idx1+1], signal1[start_idx1:end_idx1+1]

        start_idx2 = np.searchsorted(time2, start_threshold)
        end_idx2 = np.searchsorted(time2, time2[-1]-end_threshold)-1
        time_filtered2, signal_filtered2 = time2[start_idx2:end_idx2+1], signal2[start_idx2:end_idx2+1]

        # Analyse FFT
        freqs_pos1, magnitude_pos1, fundamental_frequency1, harmonics1, noise_power1, SNR1, THD1, amp_fund1, score_global1 = analyze_signal(time_filtered1, signal_filtered1, fixed_fundamental)
        freqs_pos2, magnitude_pos2, fundamental_frequency2, harmonics2, noise_power2, SNR2, THD2, amp_fund2, score_global2 = analyze_signal(time_filtered2, signal_filtered2, fixed_fundamental)

        # Signal idéal et comparaison
        signal_ideal1 = generate_ideal_signal(time_filtered1, signal_filtered1, mode='mean')
        signal_ideal2 = generate_ideal_signal(time_filtered2, signal_filtered2, mode='mean')
        diff1, rmse1, freqs_diff1, mag_diff1, noise_power_diff1 = compare_to_model(time_filtered1, signal_filtered1, signal_ideal1)
        diff2, rmse2, freqs_diff2, mag_diff2, noise_power_diff2 = compare_to_model(time_filtered2, signal_filtered2, signal_ideal2)

        # --- Graphiques FFT et harmoniques ---
        fig, axes = plt.subplots(2,2, figsize=(12,10))
        # Signal 1
        axes[0,0].plot(time_filtered1, signal_filtered1)
        axes[0,0].set_title("Signal temporel 1")
        axes[0,1].stem(freqs_pos1, magnitude_pos1, basefmt=" ")
        axes[0,1].set_xlim(0,freq_display_max)
        axes[0,1].set_title("FFT - Signal 1")
        axes[0,1].plot(fundamental_frequency1, harmonics1[0][2], 'ro', label='Fondamental')
        for h in harmonics1[1:]:
            if h[1] <= freq_display_max:
                axes[0,1].plot(h[1], h[2], 'bo')
                axes[0,1].text(h[1], h[2], f"{int(h[0])}", ha='center', va='bottom', fontsize=8)
        axes[0,1].legend()
        # Signal 2
        axes[1,0].plot(time_filtered2, signal_filtered2, color='orange')
        axes[1,0].set_title("Signal temporel 2")
        axes[1,1].stem(freqs_pos2, magnitude_pos2, basefmt=" ", linefmt='orange')
        axes[1,1].set_xlim(0,freq_display_max)
        axes[1,1].set_title("FFT - Signal 2")
        axes[1,1].plot(fundamental_frequency2, harmonics2[0][2], 'ro', label='Fondamental')
        for h in harmonics2[1:]:
            if h[1] <= freq_display_max:
                axes[1,1].plot(h[1], h[2], 'bo')
                axes[1,1].text(h[1], h[2], f"{int(h[0])}", ha='center', va='bottom', fontsize=8)
        axes[1,1].legend()
        plt.tight_layout()
        st.pyplot(fig)

        # --- Visualisation signal réel vs idéal ---
        fig2, axes2 = plt.subplots(2,1, figsize=(12,6))
        axes2[0].plot(time_filtered1, signal_filtered1, label='Signal réel 1')
        axes2[0].plot(time_filtered1, signal_ideal1, '--', label='Signal idéal 1')
        axes2[0].set_title(f"Signal réel vs idéal - Signal 1 (RMSE={rmse1:.4f})")
        axes2[0].legend()
        axes2[1].plot(time_filtered2, signal_filtered2, label='Signal réel 2', color='orange')
        axes2[1].plot(time_filtered2, signal_ideal2, '--', label='Signal idéal 2', color='green')
        axes2[1].set_title(f"Signal réel vs idéal - Signal 2 (RMSE={rmse2:.4f})")
        axes2[1].legend()
        st.pyplot(fig2)

        # --- Résultats ---
        st.write("### Paramètres et indicateurs")
        for i, (fund, SNRv, THDv, noise, harms, amp, score, rmse) in enumerate([
            (fundamental_frequency1, SNR1, THD1, noise_power1, harmonics1, amp_fund1, score_global1, rmse1),
            (fundamental_frequency2, SNR2, THD2, noise_power2, harmonics2, amp_fund2, score_global2, rmse2)
        ], start=1):
            st.write(f"**Signal {i}** :")
            st.write(f"Fréquence fondamentale = {fund:.4f} Hz")
            st.write(f"Amplitude fondamentale = {amp:.4f}")
            st.write
