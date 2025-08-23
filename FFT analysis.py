import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

st.title("Application d'analyse FFT de deux signaux")

# --- Paramètres utilisateurs visibles dès le lancement ---
start_threshold = st.number_input("Exclure les N premières secondes :", min_value=0.0, value=30.0, step=1.0)
end_threshold = st.number_input("Exclure les N dernières secondes :", min_value=0.0, value=20.0, step=1.0)

fixed_fundamental = st.number_input(
    "Forcer la fréquence fondamentale (Hz, mettre 0 pour auto)",
    min_value=0.0, value=0.0, step=1.0
)

# --- Upload CSVs ---
uploaded_file1 = st.file_uploader("Chargez le premier fichier CSV", type=["csv"])
uploaded_file2 = st.file_uploader("Chargez le deuxième fichier CSV", type=["csv"])

# --- Initialize variables ---
time_filtered1, signal_filtered1 = np.array([]), np.array([])
time_filtered2, signal_filtered2 = np.array([]), np.array([])
fundamental_frequency1 = fundamental_frequency2 = 0
harmonics1, harmonics2 = [], []
freqs_pos1, freqs_pos2 = np.array([]), np.array([])
magnitude_pos1, magnitude_pos2 = np.array([]), np.array([])
noise_power1 = noise_power2 = 0
SNR1 = SNR2 = THD1 = THD2 = 0
comparison_result = "Aucune comparaison n'a pu être effectuée."
best_signal = "Non déterminé"

# --- Vérification des fichiers uploadés ---
if uploaded_file1 is not None and uploaded_file2 is not None:
    try:
        # --- Read CSVs ---
        df1 = pd.read_csv(uploaded_file1, decimal=',')
        df2 = pd.read_csv(uploaded_file2, decimal=',')

        st.success("Les deux fichiers ont été chargés avec succès !")
        st.subheader("Aperçu des données - Fichier 1")
        st.dataframe(df1.head())
        st.subheader("Aperçu des données - Fichier 2")
        st.dataframe(df2.head())

        # --- Helper function for FFT analysis ---
        def analyze_signal(time, signal, fixed_fundamental=0.0):
            if len(time) < 2:
                return None

            dt = time[1] - time[0]
            signal_centered = signal - np.mean(signal)
            fft_vals = np.fft.fft(signal_centered)
            freqs = np.fft.fftfreq(len(signal_centered), d=dt)
            mask = freqs >= 0
            freqs_pos = freqs[mask]
            magnitude_pos = np.abs(fft_vals[mask]) / len(signal_centered)

            fundamental_freq = 0
            harmonics = []

            # --- Cas 1 : fréquence fixe ---
            if fixed_fundamental > 0:
                fundamental_freq = fixed_fundamental
                idx = np.argmin(np.abs(freqs_pos - fundamental_freq))
                harmonics.append((1, freqs_pos[idx], magnitude_pos[idx]))

            # --- Cas 2 : détection automatique ---
            elif len(magnitude_pos) > 1:
                idx = np.argmax(magnitude_pos[1:]) + 1
                fundamental_freq = freqs_pos[idx]
                harmonics.append((1, freqs_pos[idx], magnitude_pos[idx]))

            # --- Calcul 10 premiers harmoniques ---
            if fundamental_freq > 0:
                for n in range(2, 11):
                    target_freq = n * fundamental_freq
                    idx = np.argmin(np.abs(freqs_pos - target_freq))
                    harmonics.append((n, freqs_pos[idx], magnitude_pos[idx]))

            # --- Bruit ---
            noise_power = 0
            for f, m in zip(freqs_pos, magnitude_pos):
                if 0 <= f <= 10 and all(abs(f - h[1]) > 1e-6 for h in harmonics):
                    noise_power += m**2

            # --- SNR et THD ---
            if fundamental_freq > 0:
                power_fund = harmonics[0][2]**2
                power_harmonics = sum([h[2]**2 for h in harmonics[1:]])
                SNR = 10 * np.log10(power_fund / noise_power) if noise_power > 0 else np.inf
                THD = 10 * np.log10(power_harmonics / power_fund) if power_fund > 0 else -np.inf
            else:
                SNR, THD = 0, 0

            return freqs_pos, magnitude_pos, fundamental_freq, harmonics, noise_power, SNR, THD

        # --- Traitement Signal 1 ---
        if 'Time' in df1.columns and 'Signal' in df1.columns:
            time1, signal1 = df1['Time'].values, df1['Signal'].values
            t_start, t_end = start_threshold, time1[-1] - end_threshold
            start_idx, end_idx = np.argmax(time1 >= t_start), len(time1) - 1 - np.argmax(time1[::-1] <= t_end)
            if end_idx >= start_idx:
                time_filtered1, signal_filtered1 = time1[start_idx:end_idx+1], signal1[start_idx:end_idx+1]
                result = analyze_signal(time_filtered1, signal_filtered1, fixed_fundamental)
                if result:
                    freqs_pos1, magnitude_pos1, fundamental_frequency1, harmonics1, noise_power1, SNR1, THD1 = result
        else:
            st.error("Le fichier CSV du Signal 1 doit contenir les colonnes 'Time' et 'Signal'.")

        # --- Traitement Signal 2 ---
        if 'Time' in df2.columns and 'Signal' in df2.columns:
            time2, signal2 = df2['Time'].values, df2['Signal'].values
            t_start, t_end = start_threshold, time2[-1] - end_threshold
            start_idx, end_idx = np.argmax(time2 >= t_start), len(time2) - 1 - np.argmax(time2[::-1] <= t_end)
            if end_idx >= start_idx:
                time_filtered2, signal_filtered2 = time2[start_idx:end_idx+1], signal2[start_idx:end_idx+1]
                result = analyze_signal(time_filtered2, signal_filtered2, fixed_fundamental)
                if result:
                    freqs_pos2, magnitude_pos2, fundamental_frequency2, harmonics2, noise_power2, SNR2, THD2 = result
        else:
            st.error("Le fichier CSV du Signal 2 doit contenir les colonnes 'Time' et 'Signal'.")

        # --- Comparaison ---
        if fundamental_frequency1 > 0 and fundamental_frequency2 > 0:
            if SNR1 > SNR2 and THD1 < THD2:
                comparison_result = "✅ Signal 1 est globalement moins perturbé (meilleur SNR et THD plus faible)."
                best_signal = "Signal 1"
            elif SNR2 > SNR1 and THD2 < THD1:
                comparison_result = "✅ Signal 2 est globalement moins perturbé (meilleur SNR et THD plus faible)."
                best_signal = "Signal 2"
            else:
                comparison_result = "⚖️ SNR et THD comparables entre les deux signaux."
                best_signal = "Égalité"

        # --- Affichage ---
        st.subheader("Résultats FFT et indicateurs")

        # Tracés
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        if len(time_filtered1) > 1:
            axes[0,0].plot(time_filtered1, signal_filtered1)
            axes[0,0].set_title("Signal temporel 1")
            axes[0,0].grid(True)
        if len(freqs_pos1) > 0:
            axes[0,1].stem(freqs_pos1, magnitude_pos1, basefmt=" ")
            axes[0,1].set_xlim(0, 10)
            axes[0,1].set_title("Spectre FFT - Signal 1")
        if len(time_filtered2) > 1:
            axes[1,0].plot(time_filtered2, signal_filtered2, color='orange')
            axes[1,0].set_title("Signal temporel 2")
            axes[1,0].grid(True)
        if len(freqs_pos2) > 0:
            axes[1,1].stem(freqs_pos2, magnitude_pos2, basefmt=" ", linefmt='orange')
            axes[1,1].set_xlim(0, 10)
            axes[1,1].set_title("Spectre FFT - Signal 2")
        plt.tight_layout()
        st.pyplot(fig)

        # Indicateurs
        st.write(f"**Signal 1 :** Fréquence fondamentale = {fundamental_frequency1:.4f} Hz, SNR = {SNR1:.2f} dB, THD = {THD1:.2f} dB, Bruit = {noise_power1:.4f}")
        st.write(f"**Signal 2 :** Fréquence fondamentale = {fundamental_frequency2:.4f} Hz, SNR = {SNR2:.2f} dB, THD = {THD2:.2f} dB, Bruit = {noise_power2:.4f}")

        # Tableaux des harmoniques
        if harmonics1:
            st.write("#### Harmoniques - Signal 1")
            harmo_df1 = pd.DataFrame(harmonics1, columns=["Ordre", "Fréquence (Hz)", "Amplitude"])
            st.dataframe(harmo_df1)

        if harmonics2:
            st.write("#### Harmoniques - Signal 2")
            harmo_df2 = pd.DataFrame(harmonics2, columns=["Ordre", "Fréquence (Hz)", "Amplitude"])
            st.dataframe(harmo_df2)

        # --- Explications des termes ---
        st.subheader("📘 Définitions des termes utilisés")

        st.markdown("""
        - **SNR (Signal-to-Noise Ratio / Rapport Signal sur Bruit)** :  
          C'est le rapport entre la puissance de la fréquence fondamentale et la puissance du bruit présent dans le signal.  
          - **Valeurs élevées de SNR** : le signal est dominant par rapport au bruit → meilleure qualité.  
          - **Valeurs faibles de SNR** : le bruit est important par rapport au signal → signal plus perturbé.
        """)

        st.markdown("""
        - **THD (Total Harmonic Distortion / Distorsion Harmonique Totale)** :  
          Elle mesure la puissance cumulée des harmoniques (multiples entiers de la fondamentale) par rapport à la puissance de la fréquence fondamentale.  
          - **THD faible** : le signal est proche d'une sinusoïde pure → mouvement précis et stable.  
          - **THD élevée** : le signal contient beaucoup de composantes harmoniques → mouvement moins régulier et risque de vibrations.
        """)

        st.markdown("""
        - **Bruit** :  
          Puissance du signal qui n'appartient ni à la fréquence fondamentale ni à ses harmoniques importantes.  
          - **Bruit faible** : le signal est propre et fiable.  
          - **Bruit élevé** : perturbations aléatoires qui peuvent provoquer des erreurs ou des mouvements instables dans le système.
        """)

        st.write("### Conclusion de la comparaison")
        st.write(comparison_result)
        st.write(f"➡️ **Signal le plus propre : {best_signal}**")

    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {e}")

else:
    st.info("Veuillez télécharger les deux fichiers CSV pour commencer l'analyse.")
