import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import openpyxl
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter

st.title("Application d'analyse FFT de deux signaux")

uploaded_file1 = st.file_uploader("Chargez le premier fichier CSV", type=["csv"])
uploaded_file2 = st.file_uploader("Chargez le deuxième fichier CSV", type=["csv"])

start_threshold = st.number_input("Exclure les N premières secondes :", min_value=0.0, value=30.0, step=1.0)
end_threshold = st.number_input("Exclure les N dernières secondes :", min_value=0.0, value=20.0, step=1.0)

# Variables initiales
df1 = df2 = None
time_filtered1 = signal_filtered1 = np.array([])
time_filtered2 = signal_filtered2 = np.array([])
fundamental_frequency1 = fundamental_frequency2 = 0
freqs_pos1 = freqs_pos2 = np.array([])
magnitude_pos1 = magnitude_pos2 = np.array([])
noise_power1 = noise_power2 = 0
SNR1 = SNR2 = THD1 = THD2 = 0
comparison_result = "Aucune comparaison n'a pu être effectuée."
best_signal = "Non déterminé"

if uploaded_file1 is not None and uploaded_file2 is not None:
    try:
        # Lecture CSV
        df1 = pd.read_csv(uploaded_file1, decimal=',')
        df2 = pd.read_csv(uploaded_file2, decimal=',')

        st.success("Les deux fichiers ont été chargés avec succès !")
        st.subheader("Aperçu des données - Fichier 1")
        st.dataframe(df1.head())
        st.subheader("Aperçu des données - Fichier 2")
        st.dataframe(df2.head())

        # --- Fonction FFT et analyse ---
        def analyze_signal(time, signal):
            if len(time) < 2:
                return None
            dt = time[1] - time[0]
            signal_centered = signal - np.mean(signal)
            fft_vals = np.fft.fft(signal_centered)
            freqs = np.fft.fftfreq(len(signal_centered), d=dt)
            mask = freqs >= 0
            freqs_pos = freqs[mask]
            magnitude_pos = np.abs(fft_vals[mask]) / len(signal_centered)

            # Fondamentale
            fundamental_freq = 0
            if len(magnitude_pos) > 1:
                fundamental_index = np.argmax(magnitude_pos[1:]) + 1
                fundamental_freq = freqs_pos[fundamental_index]

            # Puissance du bruit [0-10 Hz] hors fondamentale
            noise_power = 0
            for f, m in zip(freqs_pos, magnitude_pos):
                if 0 <= f <= 10 and abs(f - fundamental_freq) > 1e-9:
                    noise_power += m**2

            # SNR et THD
            if fundamental_freq != 0:
                fundamental_index = np.argmin(np.abs(freqs_pos - fundamental_freq))
                power_fund = magnitude_pos[fundamental_index]**2
                power_harmonics = sum([m**2 for (f, m) in zip(freqs_pos, magnitude_pos) if f > 0 and abs(f - fundamental_freq) > 1e-9])
                SNR = 10 * np.log10(power_fund / noise_power) if noise_power > 0 else np.inf
                THD = 10 * np.log10(power_harmonics / power_fund) if power_fund > 0 else -np.inf
            else:
                SNR, THD = 0, 0

            return freqs_pos, magnitude_pos, fundamental_freq, noise_power, SNR, THD

        # --- Fonction extraction harmoniques ---
        def extract_harmonics(freqs, magnitudes, f0, n_harmonics=5):
            harmonics_data = []
            if f0 <= 0:
                return pd.DataFrame()
            for k in range(1, n_harmonics+1):
                target_freq = k * f0
                idx = np.argmin(np.abs(freqs - target_freq))
                freq_val = freqs[idx]
                amp_val = magnitudes[idx]
                rel_db = 20*np.log10(amp_val / magnitudes[np.argmax(magnitudes)]) if amp_val > 0 else -np.inf
                harmonics_data.append([k, freq_val, amp_val, rel_db])
            return pd.DataFrame(harmonics_data, columns=["Harmonique (k)", "Fréquence (Hz)", "Amplitude", "Relatif (dB)"])

        # --- Analyse Signal 1 ---
        if 'Time' in df1.columns and 'Signal' in df1.columns:
            time1, signal1 = df1['Time'].values, df1['Signal'].values
            t_start, t_end = start_threshold, time1[-1] - end_threshold
            start_idx, end_idx = np.argmax(time1 >= t_start), len(time1) - 1 - np.argmax(time1[::-1] <= t_end)
            if end_idx >= start_idx:
                time_filtered1, signal_filtered1 = time1[start_idx:end_idx+1], signal1[start_idx:end_idx+1]
                result = analyze_signal(time_filtered1, signal_filtered1)
                if result:
                    freqs_pos1, magnitude_pos1, fundamental_frequency1, noise_power1, SNR1, THD1 = result
        else:
            st.error("Le fichier CSV du Signal 1 doit contenir 'Time' et 'Signal'.")

        # --- Analyse Signal 2 ---
        if 'Time' in df2.columns and 'Signal' in df2.columns:
            time2, signal2 = df2['Time'].values, df2['Signal'].values
            t_start, t_end = start_threshold, time2[-1] - end_threshold
            start_idx, end_idx = np.argmax(time2 >= t_start), len(time2) - 1 - np.argmax(time2[::-1] <= t_end)
            if end_idx >= start_idx:
                time_filtered2, signal_filtered2 = time2[start_idx:end_idx+1], signal2[start_idx:end_idx+1]
                result = analyze_signal(time_filtered2, signal_filtered2)
                if result:
                    freqs_pos2, magnitude_pos2, fundamental_frequency2, noise_power2, SNR2, THD2 = result
        else:
            st.error("Le fichier CSV du Signal 2 doit contenir 'Time' et 'Signal'.")

        # --- Extraction harmoniques ---
        harmonics_df1 = extract_harmonics(freqs_pos1, magnitude_pos1, fundamental_frequency1)
        harmonics_df2 = extract_harmonics(freqs_pos2, magnitude_pos2, fundamental_frequency2)

        # --- Comparaison SNR + THD ---
        if len(freqs_pos1) > 0 and len(freqs_pos2) > 0:
            if SNR1 > SNR2 and THD1 < THD2:
                comparison_result = "✅ Signal 1 est globalement moins perturbé."
                best_signal = "Signal 1"
            elif SNR2 > SNR1 and THD2 < THD1:
                comparison_result = "✅ Signal 2 est globalement moins perturbé."
                best_signal = "Signal 2"
            else:
                comparison_result = "⚖️ Les deux signaux ont des compromis différents."
                best_signal = "Égalité"

        # --- Graphiques ---
        st.subheader("Analyse graphique")
        if len(freqs_pos1) > 0 or len(freqs_pos2) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            if len(time_filtered1) > 1:
                axes[0,0].plot(time_filtered1, signal_filtered1, label="Signal 1")
                axes[0,0].set_title("Signal temporel 1")
            if len(freqs_pos1) > 0:
                axes[0,1].stem(freqs_pos1, magnitude_pos1, basefmt=" ")
                axes[0,1].set_xlim(0,10)
                axes[0,1].set_title("Spectre FFT - Signal 1")
            if len(time_filtered2) > 1:
                axes[1,0].plot(time_filtered2, signal_filtered2, color='orange')
                axes[1,0].set_title("Signal temporel 2")
            if len(freqs_pos2) > 0:
                axes[1,1].stem(freqs_pos2, magnitude_pos2, basefmt=" ", linefmt='orange')
                axes[1,1].set_xlim(0,10)
                axes[1,1].set_title("Spectre FFT - Signal 2")
            plt.tight_layout()
            st.pyplot(fig)

        # --- Résultats numériques ---
        st.subheader("Résultats numériques")
        st.write(f"**Signal 1 :** f₀ = {fundamental_frequency1:.4f} Hz, "
                 f"S
