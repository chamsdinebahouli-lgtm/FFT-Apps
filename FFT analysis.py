import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import io

st.title("Application d'analyse FFT de deux signaux avec rapport PDF")

# --- Paramètres ---
start_threshold = st.number_input("Exclure les N premières secondes :", min_value=0.0, value=30.0, step=1.0)
end_threshold = st.number_input("Exclure les N dernières secondes :", min_value=0.0, value=20.0, step=1.0)
fixed_fundamental = st.number_input("Forcer la fréquence fondamentale (Hz, mettre 0 pour auto)", min_value=0.0, value=0.0, step=1.0)

# --- Upload CSVs ---
uploaded_file1 = st.file_uploader("Chargez le premier fichier CSV", type=["csv"])
uploaded_file2 = st.file_uploader("Chargez le deuxième fichier CSV", type=["csv"])

# --- Fonction d'analyse FFT ---
def analyze_signal(time, signal, fixed_fundamental=0.0):
    dt = time[1]-time[0]
    sig_centered = signal - np.mean(signal)
    fft_vals = np.fft.fft(sig_centered)
    freqs = np.fft.fftfreq(len(sig_centered), d=dt)
    mask = freqs >= 0
    freqs_pos = freqs[mask]
    magnitude_pos = np.abs(fft_vals[mask])/len(sig_centered)

    fundamental_freq = 0
    harmonics = []

    if fixed_fundamental > 0:
        fundamental_freq = fixed_fundamental
        idx = np.argmin(np.abs(freqs_pos - fundamental_freq))
        harmonics.append((1, freqs_pos[idx], magnitude_pos[idx]))
    elif len(magnitude_pos) > 1:
        idx = np.argmax(magnitude_pos[1:]) + 1
        fundamental_freq = freqs_pos[idx]
        harmonics.append((1, freqs_pos[idx], magnitude_pos[idx]))

    # 10 premiers harmoniques
    if fundamental_freq > 0:
        for n in range(2, 11):
            target = n*fundamental_freq
            idx = np.argmin(np.abs(freqs_pos - target))
            harmonics.append((n, freqs_pos[idx], magnitude_pos[idx]))

    # Bruit
    noise_power = sum([m**2 for f,m in zip(freqs_pos, magnitude_pos) if 0<=f<=10 and all(abs(f-h[1])>1e-6 for h in harmonics)])
    power_fund = harmonics[0][2]**2 if harmonics else 0
    power_harmo = sum([h[2]**2 for h in harmonics[1:]]) if harmonics else 0
    SNR = 10*np.log10(power_fund/noise_power) if noise_power>0 else np.inf
    THD = 10*np.log10(power_harmo/power_fund) if power_fund>0 else -np.inf

    return freqs_pos, magnitude_pos, fundamental_freq, harmonics, noise_power, SNR, THD

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
        freqs_pos1, magnitude_pos1, fundamental_frequency1, harmonics1, noise_power1, SNR1, THD1 = analyze_signal(time_filtered1, signal_filtered1, fixed_fundamental)

        t_start2, t_end2 = start_threshold, time2[-1]-end_threshold
        start_idx2, end_idx2 = np.argmax(time2>=t_start2), len(time2)-1 - np.argmax(time2[::-1]<=t_end2)
        time_filtered2, signal_filtered2 = time2[start_idx2:end_idx2+1], signal2[start_idx2:end_idx2+1]
        freqs_pos2, magnitude_pos2, fundamental_frequency2, harmonics2, noise_power2, SNR2, THD2 = analyze_signal(time_filtered2, signal_filtered2, fixed_fundamental)

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

        # --- Comparaison ---
        if SNR1>SNR2 and THD1<THD2:
            best_signal="Signal 1"
            comparison_result="Signal 1 globalement meilleur"
        elif SNR2>SNR1 and THD2<THD1:
            best_signal="Signal 2"
            comparison_result="Signal 2 globalement meilleur"
        else:
            best_signal="Égalité"
            comparison_result="SNR et THD comparables"

        st.write(f"### Comparaison : {comparison_result} (Signal le plus propre : {best_signal})")

        # --- Bouton PDF ---
        if st.button("📄 Générer et télécharger rapport PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Rapport FFT - Analyse des signaux", ln=1, align='C')
            pdf.set_font("Arial", '', 12)
            pdf.ln(5)
            pdf.cell(0, 10, f"Paramètres : Start={start_threshold}s, End={end_threshold}s, Fréquence forcée={fixed_fundamental}Hz", ln=1)

            # Sauvegarde graphique dans un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix=".png") as tmpfile:
                fig.savefig(tmpfile.name, format='png')
                tmpfile.flush()
                pdf.image(tmpfile.name, x=10, y=40, w=190)

            # Tableau des indicateurs et harmoniques
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Indicateurs et harmoniques", ln=1)
            pdf.set_font("Arial", '', 12)
            pdf.ln(5)
            for i, (fund, SNRv, THDv, noise, harms) in enumerate([
                (fundamental_frequency1, SNR1, THD1, noise_power1, harmonics1),
                (fundamental_frequency2, SNR2, THD2, noise_power2, harmonics2)
            ], start=1):
                pdf.cell(0, 10, f"Signal {i}: Fréquence fondamentale={fund:.4f}Hz, SNR={SNRv:.2f}dB, THD={THDv:.2f}dB, Bruit={noise:.4f}", ln=1)
                pdf.cell(0, 10, "Harmoniques (Ordre, Fréquence, Amplitude):", ln=1)
                for h in harms:
                    pdf.cell(0, 8, f"{h[0]}, {h[1]:.4f} Hz, {h[2]:.4f}", ln=1)
                pdf.ln(3)

            # PDF en mémoire pour téléchargement
            pdf_buffer = io.BytesIO()
            pdf.output(pdf_buffer)
            pdf_buffer.seek(0)
            st.download_button("⬇️ Télécharger le rapport PDF", data=pdf_buffer, file_name="rapport_fft.pdf", mime="application/pdf")

    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {e}")
else:
    st.info("Veuillez télécharger les deux fichiers CSV pour commencer l'analyse.")
