import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

st.title("Application d'analyse FFT de deux signaux")

# --- Encadré explicatif ---
st.subheader("💡 Définitions des indicateurs de qualité du signal")
st.markdown("""
- **Fréquence fondamentale (Hz)** : fréquence principale du signal correspondant au mouvement moteur.
- **Amplitude fondamentale** : intensité de la fréquence principale.
- **SNR (Signal-to-Noise Ratio, dB)** : rapport entre la puissance du signal fondamental et le bruit. Plus SNR élevé → signal propre.
- **THD (Total Harmonic Distortion, dB)** : mesure de la distorsion du signal via les harmoniques. Plus THD faible → signal moins déformé.
- **Bruit (0-10 Hz hors harmoniques)** : énergie hors des harmoniques principales, indicateur de perturbations.
- **Score global** : combinaison pondérée de SNR, THD, bruit et amplitude fondamentale pour comparer la qualité des signaux.
""")

uploaded_file1 = st.file_uploader("Chargez le premier fichier CSV", type=["csv"])
uploaded_file2 = st.file_uploader("Chargez le deuxième fichier CSV", type=["csv"])

start_threshold = st.number_input("Exclure les N premières secondes :", min_value=0.0, value=30.0, step=1.0)
end_threshold = st.number_input("Exclure les N dernières secondes :", min_value=0.0, value=20.0, step=1.0)
fixed_fundamental = st.number_input("Forcer la fréquence fondamentale (Hz, mettre 0 pour auto)", min_value=0.0, value=0.0, step=1.0)

# --- Fonction d'analyse FFT ---
def analyze_signal(time, signal, fixed_fundamental=0.0):
    dt = time[1] - time[0]
    sig_centered = signal - np.mean(signal)
    fft_vals = np.fft.fft(sig_centered)
    freqs = np.fft.fftfreq(len(sig_centered), d=dt)
    mask = freqs >= 0

    # CORRECTION : multiplication par 2 pour amplitudes unilatérales
    magnitude_pos = 2 * np.abs(fft_vals[mask]) / len(sig_centered)
    freqs_pos = freqs[mask]

    # Trouver la fréquence fondamentale
    fundamental_freq = 0
    harmonics = []

    if fixed_fundamental > 0:
        fundamental_freq = fixed_fundamental
        # Interpolation pour amplitude exacte
        magnitude_interp = np.interp(fundamental_freq, freqs_pos, magnitude_pos)
        harmonics.append((1, fundamental_freq, magnitude_interp))
    elif len(magnitude_pos) > 1:
        idx = np.argmax(magnitude_pos[1:]) + 1
        fundamental_freq = freqs_pos[idx]
        harmonics.append((1, fundamental_freq, magnitude_pos[idx]))

    # 10 premiers harmoniques
    if fundamental_freq > 0:
        for n in range(2, 11):
            target = n * fundamental_freq
            if target <= freqs_pos[-1]:
                magnitude_interp = np.interp(target, freqs_pos, magnitude_pos)
                harmonics.append((n, target, magnitude_interp))

    # Bruit (0-10 Hz, hors harmoniques)
    noise_power = sum([m**2 for f,m in zip(freqs_pos, magnitude_pos) if 0<=f<=10 and all(abs(f-h[1])>1e-6 for h in harmonics)])

    power_fund = harmonics[0][2]**2 if harmonics else 0
    power_harmo = sum([h[2]**2 for h in harmonics[1:]]) if harmonics else 0

    SNR = 10*np.log10(power_fund / noise_power) if noise_power>0 else np.inf
    THD = 10*np.log10(power_harmo / power_fund) if power_fund>0 else -np.inf
    amp_fundamental = harmonics[0][2] if harmonics else 0

    # Score global combiné
    score_global = 0.4*SNR - 0.3*THD - 0.2*noise_power + 0.1*amp_fundamental

    return freqs_pos, magnitude_pos, fundamental_freq, harmonics, noise_power, SNR, THD, amp_fundamental, score_global

# --- Traitement des fichiers ---
if uploaded_file1 and uploaded_file2:
    try:
        df1 = pd.read_csv(uploaded_file1, decimal=',')
        df2 = pd.read_csv(uploaded_file2, decimal=',')

        time1, signal1 = df1['Time'].values, df1['Signal'].values
        time2, signal2 = df2['Time'].values, df2['Signal'].values

        # Filtrage temporel
        t_start1, t_end1 = start_threshold, time1[-1]-end_threshold
        start_idx1, end_idx1 = np.argmax(time1>=t_start1), len(time1)-1 - np.argmax(time1[::-1]<=t_end1)
        time_filtered1, signal_filtered1 = time1[start_idx1:end_idx1+1], signal1[start_idx1:end_idx1+1]
        freqs_pos1, magnitude_pos1, fundamental_frequency1, harmonics1, noise_power1, SNR1, THD1, amp_fund1, score_global1 = analyze_signal(time_filtered1, signal_filtered1, fixed_fundamental)

        t_start2, t_end2 = start_threshold, time2[-1]-end_threshold
        start_idx2, end_idx2 = np.argmax(time2>=t_start2), len(time2)-1 - np.argmax(time2[::-1]<=t_end2)
        time_filtered2, signal_filtered2 = time2[start_idx2:end_idx2+1], signal2[start_idx2:end_idx2+1]
        freqs_pos2, magnitude_pos2, fundamental_frequency2, harmonics2, noise_power2, SNR2, THD2, amp_fund2, score_global2 = analyze_signal(time_filtered2, signal_filtered2, fixed_fundamental)

        # --- Affichage graphique ---
        fig, axes = plt.subplots(2,2, figsize=(12,10))
        axes[0,0].plot(time_filtered1, signal_filtered1)
        axes[0,0].set_title("Signal temporel 1")
        axes[0,1].stem(freqs_pos1, magnitude_pos1, basefmt=" ")
        axes[0,1].set_xlim(0,10)
        axes[0,1].set_title("FFT - Signal 1")
        axes[1,0].plot(time_filtered2, signal_filtered2, color='orange')
        axes[1,0].set_title("Signal temporel 2")
        axes[1,1].stem(freqs_pos2, magnitude_pos2, basefmt=" ", linefmt='orange')
        axes[1,1].set_xlim(0,10)
        axes[1,1].set_title("FFT - Signal 2")
        plt.tight_layout()
        st.pyplot(fig)

        # --- Affichage des résultats ---
        st.write("### Paramètres et indicateurs")
        for i, (fund, SNRv, THDv, noise, harms, amp, score) in enumerate([
            (fundamental_frequency1, SNR1, THD1, noise_power1, harmonics1, amp_fund1, score_global1),
            (fundamental_frequency2, SNR2, THD2, noise_power2, harmonics2, amp_fund2, score_global2)
        ], start=1):
            st.write(f"**Signal {i}** :")
            st.write(f"Fréquence fondamentale = {fund:.4f} Hz")
            st.write(f"Amplitude fondamentale = {amp:.4f}")
            st.write(f"SNR = {SNRv:.2f} dB")
            st.write(f"THD = {THDv:.2f} dB")
            st.write(f"Bruit (0-10 Hz hors harmoniques) = {noise:.4f}")
            st.write(f"Score global = {score:.2f}")
            st.write("Harmoniques :")
            harms_df = pd.DataFrame(harms, columns=["Ordre", "Fréquence (Hz)", "Amplitude"])
            st.dataframe(harms_df)

        # --- Comparaison ---
        if score_global1 > score_global2:
            best_signal="Signal 1"
            comparison_result="Signal 1 globalement meilleur selon le score combiné"
        elif score_global2 > score_global1:
            best_signal="Signal 2"
            comparison_result="Signal 2 globalement meilleur selon le score combiné"
        else:
            best_signal="Égalité"
            comparison_result="Les deux signaux sont équivalents selon le score combiné"

        st.write("### Comparaison globale")
        st.write(f"Signal le plus propre : {best_signal}")
        st.write(comparison_result)

        # --- Export CSV des résultats et harmoniques ---
        all_data = []
        for i, (fund, SNRv, THDv, noise, harms, amp, score) in enumerate([
            (fundamental_frequency1, SNR1, THD1, noise_power1, harmonics1, amp_fund1, score_global1),
            (fundamental_frequency2, SNR2, THD2, noise_power2, harmonics2, amp_fund2, score_global2)
        ], start=1):
            for order, freq, magnitude in harms:
                all_data.append({
                    "Signal": f"Signal {i}",
                    "Ordre harmonique": order,
                    "Fréquence (Hz)": freq,
                    "Amplitude": magnitude,
                    "Fréquence fondamentale (Hz)": fund,
                    "Amplitude fondamentale": amp,
                    "SNR (dB)": SNRv,
                    "THD (dB)": THDv,
                    "Bruit (0-10Hz)": noise,
                    "Score global": score
                })

        if all_data:
            harmonics_df = pd.DataFrame(all_data)
            csv_buffer = io.StringIO()
            harmonics_df.to_csv(csv_buffer, index=False, sep=";")
            st.download_button(
                label="📥 Télécharger toutes les données des harmoniques et paramètres (CSV)",
                data=csv_buffer.getvalue(),
                file_name="harmoniques_signaux.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {e}")
else:
    st.info("Veuillez télécharger les deux fichiers CSV pour commencer l'analyse.")
