import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

st.title("Application d'analyse FFT de deux signaux")

uploaded_file1 = st.file_uploader("Chargez le premier fichier CSV", type=["csv"])
uploaded_file2 = st.file_uploader("Chargez le deuxième fichier CSV", type=["csv"])

start_threshold = st.number_input("Exclure les N premières secondes :", min_value=0.0, value=30.0, step=1.0)
end_threshold = st.number_input("Exclure les N dernières secondes :", min_value=0.0, value=20.0, step=1.0)

# Initialize variables
df1 = None
df2 = None
time_filtered1, signal_filtered1 = np.array([]), np.array([])
time_filtered2, signal_filtered2 = np.array([]), np.array([])
fundamental_frequency1 = fundamental_frequency2 = 0
prominent_freqs1, prominent_freqs2 = [], []
freqs_pos1, freqs_pos2 = np.array([]), np.array([])
magnitude_pos1, magnitude_pos2 = np.array([]), np.array([])
noise_power1 = noise_power2 = 0
SNR1 = SNR2 = THD1 = THD2 = 0
comparison_result = "Aucune comparaison n'a pu être effectuée."
best_signal = "Non déterminé"

if uploaded_file1 is not None and uploaded_file2 is not None:
    try:
        # Read CSVs
        df1 = pd.read_csv(uploaded_file1, decimal=',')
        df2 = pd.read_csv(uploaded_file2, decimal=',')

        st.success("Les deux fichiers ont été chargés avec succès !")
        st.subheader("Aperçu des données - Fichier 1")
        st.dataframe(df1.head())
        st.subheader("Aperçu des données - Fichier 2")
        st.dataframe(df2.head())

        # --- Helper function for FFT analysis ---
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

            # Trouver la fondamentale
            fundamental_freq = 0
            prominent_freqs = []
            if len(magnitude_pos) > 1:
                fundamental_index = np.argmax(magnitude_pos[1:]) + 1
                fundamental_freq = freqs_pos[fundamental_index]
                prominent_freqs.append((fundamental_freq, magnitude_pos[fundamental_index]))

            # Ajouter quelques harmoniques proéminentes
            sorted_indices = np.argsort(magnitude_pos[1:])[::-1] + 1
            for i in sorted_indices[:5]:
                freq, mag = freqs_pos[i], magnitude_pos[i]
                if abs(freq - fundamental_freq) > 1e-9:
                    prominent_freqs.append((freq, mag))

            # Puissance de bruit dans [0-10 Hz], hors fondamentale
            noise_power = 0
            for f, m in zip(freqs_pos, magnitude_pos):
                if 0 <= f <= 10 and abs(f - fundamental_freq) > 1e-9:
                    noise_power += m**2

            # Calcul SNR + THD
            if fundamental_freq != 0:
                fundamental_index = np.argmin(np.abs(freqs_pos - fundamental_freq))
                power_fund = magnitude_pos[fundamental_index]**2
                power_harmonics = sum([mag**2 for (freq, mag) in prominent_freqs if abs(freq - fundamental_freq) > 1e-9])

                SNR = 10 * np.log10(power_fund / noise_power) if noise_power > 0 else np.inf
                THD = 10 * np.log10(power_harmonics / power_fund) if power_fund > 0 else -np.inf
            else:
                SNR, THD = 0, 0

            return freqs_pos, magnitude_pos, fundamental_freq, prominent_freqs, noise_power, SNR, THD

        # --- Process Signal 1 ---
        if 'Time' in df1.columns and 'Signal' in df1.columns:
            time1, signal1 = df1['Time'].values, df1['Signal'].values
            t_start, t_end = start_threshold, time1[-1] - end_threshold
            start_idx, end_idx = np.argmax(time1 >= t_start), len(time1) - 1 - np.argmax(time1[::-1] <= t_end)
            if end_idx >= start_idx:
                time_filtered1, signal_filtered1 = time1[start_idx:end_idx+1], signal1[start_idx:end_idx+1]
                result = analyze_signal(time_filtered1, signal_filtered1)
                if result:
                    freqs_pos1, magnitude_pos1, fundamental_frequency1, prominent_freqs1, noise_power1, SNR1, THD1 = result
        else:
            st.error("Le fichier CSV du Signal 1 doit contenir les colonnes 'Time' et 'Signal'.")

        # --- Process Signal 2 ---
        if 'Time' in df2.columns and 'Signal' in df2.columns:
            time2, signal2 = df2['Time'].values, df2['Signal'].values
            t_start, t_end = start_threshold, time2[-1] - end_threshold
            start_idx, end_idx = np.argmax(time2 >= t_start), len(time2) - 1 - np.argmax(time2[::-1] <= t_end)
            if end_idx >= start_idx:
                time_filtered2, signal_filtered2 = time2[start_idx:end_idx+1], signal2[start_idx:end_idx+1]
                result = analyze_signal(time_filtered2, signal_filtered2)
                if result:
                    freqs_pos2, magnitude_pos2, fundamental_frequency2, prominent_freqs2, noise_power2, SNR2, THD2 = result
        else:
            st.error("Le fichier CSV du Signal 2 doit contenir les colonnes 'Time' et 'Signal'.")

        # --- Comparaison basée sur SNR + THD ---
        if len(freqs_pos1) > 0 and len(freqs_pos2) > 0:
            if SNR1 > SNR2 and THD1 < THD2:
                comparison_result = "✅ Signal 1 est globalement moins perturbé (meilleur SNR et plus faible THD)."
                best_signal = "Signal 1"
            elif SNR2 > SNR1 and THD2 < THD1:
                comparison_result = "✅ Signal 2 est globalement moins perturbé (meilleur SNR et plus faible THD)."
                best_signal = "Signal 2"
            else:
                if SNR1 > SNR2:
                    comparison_result = "⚖️ Signal 1 a un meilleur SNR (moins de bruit), mais Signal 2 a une distorsion plus faible (THD)."
                    best_signal = "Signal 1 (si le bruit est plus critique)"
                elif SNR2 > SNR1:
                    comparison_result = "⚖️ Signal 2 a un meilleur SNR (moins de bruit), mais Signal 1 a une distorsion plus faible (THD)."
                    best_signal = "Signal 2 (si le bruit est plus critique)"
                else:
                    comparison_result = "ℹ️ Les deux signaux présentent des perturbations similaires (SNR et THD comparables)."
                    best_signal = "Égalité"

        # --- Affichage ---
        st.subheader("Résultats de l'analyse")

        # Tracés
        if len(freqs_pos1) > 0 or len(freqs_pos2) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            # Signal 1
            if len(time_filtered1) > 1:
                axes[0, 0].plot(time_filtered1, signal_filtered1, label="Signal 1")
                axes[0, 0].set_title("Signal temporel 1")
                axes[0, 0].grid(True)
            if len(freqs_pos1) > 0:
                axes[0, 1].stem(freqs_pos1, magnitude_pos1, basefmt=" ")
                axes[0, 1].set_xlim(0, 10)
                axes[0, 1].set_title("Spectre FFT - Signal 1")
            # Signal 2
            if len(time_filtered2) > 1:
                axes[1, 0].plot(time_filtered2, signal_filtered2, color='orange', label="Signal 2")
                axes[1, 0].set_title("Signal temporel 2")
                axes[1, 0].grid(True)
            if len(freqs_pos2) > 0:
                axes[1, 1].stem(freqs_pos2, magnitude_pos2, basefmt=" ", linefmt='orange')
                axes[1, 1].set_xlim(0, 10)
                axes[1, 1].set_title("Spectre FFT - Signal 2")
            plt.tight_layout()
            st.pyplot(fig)

        # Résultats numériques
        st.write("### Indicateurs de qualité du signal")
        st.write(f"**Signal 1 :** Fréquence fondamentale = {fundamental_frequency1:.4f} Hz, SNR = {SNR1:.2f} dB, THD = {THD1:.2f} dB, Bruit = {noise_power1:.4f}")
        st.write(f"**Signal 2 :** Fréquence fondamentale = {fundamental_frequency2:.4f} Hz, SNR = {SNR2:.2f} dB, THD = {THD2:.2f} dB, Bruit = {noise_power2:.4f}")

        # --- Affichage des harmoniques ---
        if len(prominent_freqs1) > 0:
            st.write("### Harmoniques - Signal 1")
            df_harm1 = pd.DataFrame([
                {"k": i+1, "Fréquence (Hz)": f, "Amplitude": a}
                for i, (f, a) in enumerate(prominent_freqs1)
            ])
            st.dataframe(df_harm1.style.format({"Fréquence (Hz)": "{:.4f}", "Amplitude": "{:.6f}"}))

        if len(prominent_freqs2) > 0:
            st.write("### Harmoniques - Signal 2")
            df_harm2 = pd.DataFrame([
                {"k": i+1, "Fréquence (Hz)": f, "Amplitude": a}
                for i, (f, a) in enumerate(prominent_freqs2)
            ])
            st.dataframe(df_harm2.style.format({"Fréquence (Hz)": "{:.4f}", "Amplitude": "{:.6f}"}))

        st.write("### Conclusion de la comparaison")
        st.write(comparison_result)

        st.write("### Classement automatique")
        st.write(f"➡️ **Signal le plus propre : {best_signal}**")

        # --- Export CSV des résultats ---
        results_df = pd.DataFrame({
            "Signal": ["Signal 1", "Signal 2"],
            "Fréquence_fondamentale_Hz": [fundamental_frequency1, fundamental_frequency2],
            "SNR_dB": [SNR1, SNR2],
            "THD_dB": [THD1, THD2],
            "Puissance_bruit": [noise_power1, noise_power2]
        })

        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False, sep=";")
        st.download_button(
            label="📥 Télécharger les résultats (CSV)",
            data=csv_buffer.getvalue(),
            file_name="resultats_signaux.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {e}")

else:
    st.info("Veuillez télécharger les deux fichiers CSV pour commencer l'analyse.")
