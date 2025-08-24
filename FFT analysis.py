import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

st.set_page_config(layout="wide")
st.title("Application d'analyse FFT de deux signaux")

# --- Encadré explicatif ---
st.subheader("💡 Définitions des indicateurs de qualité du signal")
st.markdown("""
- **Fréquence fondamentale (Hz)** : fréquence principale du signal correspondant au mouvement moteur.
- **Amplitude fondamentale** : intensité de la fréquence principale.
- **SNR (Signal-to-Noise Ratio, dB)** : rapport entre la puissance du signal fondamental et le bruit. Plus SNR élevé → signal propre.
- **THD (Total Harmonic Distortion, dB)** : mesure de la distorsion du signal via les harmoniques. Plus THD faible → signal moins déformé.
- **Bruit (0-10 Hz hors harmoniques)** : énergie hors des harmoniques principales, indicateur de perturbations.
- **RMSE vs signal idéal** : écart quadratique moyen entre le signal et un modèle idéal continu.
- **Score global** : combinaison pondérée de SNR, THD, bruit, amplitude fondamentale et RMSE.
- **Analyse par segments** : permet d'identifier les portions temporelles les plus proches ou les plus éloignées du modèle.
""")

# --- Sidebar pour réglages ---
st.sidebar.header("⚙️ Paramètres d'analyse")
uploaded_file1 = st.sidebar.file_uploader("Chargez le premier fichier CSV", type=["csv"])
uploaded_file2 = st.sidebar.file_uploader("Chargez le deuxième fichier CSV", type=["csv"])

start_threshold = st.sidebar.number_input("Exclure les N premières secondes :", min_value=0.0, value=30.0, step=1.0)
end_threshold = st.sidebar.number_input("Exclure les N dernières secondes :", min_value=0.0, value=20.0, step=1.0)
fixed_fundamental = st.sidebar.number_input("Forcer la fréquence fondamentale (Hz, mettre 0 pour auto)", min_value=0.0, value=0.0, step=1.0)
segment_duration = st.sidebar.number_input("Durée d’un segment (secondes)", min_value=1.0, value=10.0, step=1.0)

# --- Analyse FFT ---
def analyze_signal(time, signal, fixed_fundamental=0.0):
    dt = time[1] - time[0]
    sig_centered = signal - np.mean(signal)
    fft_vals = np.fft.fft(sig_centered)
    freqs = np.fft.fftfreq(len(sig_centered), d=dt)
    mask = freqs >= 0

    magnitude_pos = 2 * np.abs(fft_vals[mask]) / len(sig_centered)
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

    noise_power = sum([m**2 for f,m in zip(freqs_pos, magnitude_pos) if 0<=f<=10 and all(abs(f-h[1])>1e-6 for h in harmonics)])

    power_fund = harmonics[0][2]**2 if harmonics else 0
    power_harmo = sum([h[2]**2 for h in harmonics[1:]]) if harmonics else 0

    SNR = 10*np.log10(power_fund / noise_power) if noise_power>0 else np.inf
    THD = 10*np.log10(power_harmo / power_fund) if power_fund>0 else -np.inf
    amp_fundamental = harmonics[0][2] if harmonics else 0

    score_global = 0.35*SNR - 0.25*THD - 0.2*noise_power + 0.1*amp_fundamental

    return freqs_pos, magnitude_pos, fundamental_freq, harmonics, noise_power, SNR, THD, amp_fundamental, score_global

# --- Signal modèle continu ---
def compute_rmse(time, signal):
    ideal_signal = np.ones_like(signal) * np.mean(signal)
    rmse = np.sqrt(np.mean((signal - ideal_signal) ** 2))
    return rmse, ideal_signal

# --- Analyse par segments ---
def analyze_segments(time, signal, segment_duration, fixed_fundamental=0.0):
    segments = []
    dt = time[1] - time[0]
    n_points = int(segment_duration / dt)
    for i in range(0, len(time)-n_points, n_points):
        t_seg = time[i:i+n_points]
        s_seg = signal[i:i+n_points]
        if len(t_seg) < 2:
            continue
        rmse, _ = compute_rmse(t_seg, s_seg)
        _, _, f0, _, noise, SNR, THD, amp, score = analyze_signal(t_seg, s_seg, fixed_fundamental)
        segments.append({
            "t_start": t_seg[0], "t_end": t_seg[-1],
            "RMSE": rmse, "SNR": SNR, "THD": THD,
            "Bruit": noise, "Amplitude f0": amp,
            "f0": f0, "Score global": score
        })
    return pd.DataFrame(segments)

# --- Traitement fichiers ---
if uploaded_file1 and uploaded_file2:
    try:
        df1 = pd.read_csv(uploaded_file1, decimal=',')
        df2 = pd.read_csv(uploaded_file2, decimal=',')

        time1, signal1 = df1['Time'].values, df1['Signal'].values
        time2, signal2 = df2['Time'].values, df2['Signal'].values

        t_start1, t_end1 = start_threshold, time1[-1]-end_threshold
        start_idx1, end_idx1 = np.argmax(time1>=t_start1), len(time1)-1 - np.argmax(time1[::-1]<=t_end1)
        time_filtered1, signal_filtered1 = time1[start_idx1:end_idx1+1], signal1[start_idx1:end_idx1+1]

        t_start2, t_end2 = start_threshold, time2[-1]-end_threshold
        start_idx2, end_idx2 = np.argmax(time2>=t_start2), len(time2)-1 - np.argmax(time2[::-1]<=t_end2)
        time_filtered2, signal_filtered2 = time2[start_idx2:end_idx2+1], signal2[start_idx2:end_idx2+1]

        freqs_pos1, magnitude_pos1, f0_1, harmonics1, noise_power1, SNR1, THD1, amp_fund1, score_global1 = analyze_signal(time_filtered1, signal_filtered1, fixed_fundamental)
        freqs_pos2, magnitude_pos2, f0_2, harmonics2, noise_power2, SNR2, THD2, amp_fund2, score_global2 = analyze_signal(time_filtered2, signal_filtered2, fixed_fundamental)

        rmse1, ideal1 = compute_rmse(time_filtered1, signal_filtered1)
        rmse2, ideal2 = compute_rmse(time_filtered2, signal_filtered2)

        # --- Graphiques avec modèle + f0 ---
        fig, axes = plt.subplots(2,2, figsize=(12,10))
        axes[0,0].plot(time_filtered1, signal_filtered1, label="Signal 1")
        axes[0,0].plot(time_filtered1, ideal1, 'r--', label="Modèle idéal")
        axes[0,0].legend()
        axes[0,0].set_title("Signal temporel 1 (avec modèle)")

        axes[0,1].stem(freqs_pos1, magnitude_pos1, basefmt=" ")
        axes[0,1].axvline(f0_1, color="r", linestyle="--", label=f"f0={f0_1:.2f}Hz, A={amp_fund1:.2f}")
        axes[0,1].set_xlim(0,10)
        axes[0,1].legend()
        axes[0,1].set_title("FFT - Signal 1")

        axes[1,0].plot(time_filtered2, signal_filtered2, color='orange', label="Signal 2")
        axes[1,0].plot(time_filtered2, ideal2, 'r--', label="Modèle idéal")
        axes[1,0].legend()
        axes[1,0].set_title("Signal temporel 2 (avec modèle)")

        axes[1,1].stem(freqs_pos2, magnitude_pos2, basefmt=" ", linefmt='orange')
        axes[1,1].axvline(f0_2, color="r", linestyle="--", label=f"f0={f0_2:.2f}Hz, A={amp_fund2:.2f}")
        axes[1,1].set_xlim(0,10)
        axes[1,1].legend()
        axes[1,1].set_title("FFT - Signal 2")

        plt.tight_layout()
        st.pyplot(fig)

        # --- Comparaison chiffrée ---
        st.write("### Comparaison globale détaillée")
        comparison_data = {
            "Critère": ["RMSE vs modèle", "SNR (dB)", "THD (dB)", "Puissance de bruit", "Score global"],
            "Signal 1": [rmse1, SNR1, THD1, noise_power1, score_global1],
            "Signal 2": [rmse2, SNR2, THD2, noise_power2, score_global2],
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)

        # --- Graphe comparatif barres ---
        fig2, ax2 = plt.subplots(figsize=(8,5))
        ind = np.arange(len(comparison_data["Critère"]))
        width = 0.35
        ax2.bar(ind-width/2, comparison_df["Signal 1"], width, label="Signal 1")
        ax2.bar(ind+width/2, comparison_df["Signal 2"], width, label="Signal 2")
        ax2.set_xticks(ind)
        ax2.set_xticklabels(comparison_data["Critère"], rotation=30, ha="right")
        ax2.set_title("Comparaison des indicateurs")
        ax2.legend()
        st.pyplot(fig2)

        # --- Analyse automatique ---
        comments = []
        if rmse1 < rmse2:
            comments.append("➡️ **Signal 1** suit mieux le modèle idéal (RMSE plus faible).")
        else:
            comments.append("➡️ **Signal 2** suit mieux le modèle idéal (RMSE plus faible).")
        if SNR1 > SNR2:
            comments.append("➡️ **Signal 1** est moins bruité (SNR plus élevé).")
        else:
            comments.append("➡️ **Signal 2** est moins bruité (SNR plus élevé).")
        if THD1 < THD2:
            comments.append("➡️ **Signal 1** présente moins de distorsion harmonique (THD plus faible).")
        else:
            comments.append("➡️ **Signal 2** présente moins de distorsion harmonique (THD plus faible).")
        if noise_power1 < noise_power2:
            comments.append("➡️ **Signal 1** contient moins d'énergie de bruit.")
        else:
            comments.append("➡️ **Signal 2** contient moins d'énergie de bruit.")

        if score_global1 > score_global2:
            final = "✅ **Signal 1 est globalement meilleur selon le score combiné.**"
        elif score_global2 > score_global1:
            final = "✅ **Signal 2 est globalement meilleur selon le score combiné.**"
        else:
            final = "⚖️ Les deux signaux sont équivalents selon le score combiné."

        for c in comments:
            st.write(c)
        st.subheader(final)

        # --- Analyse par segments ---
        st.write("### Analyse par segments")
        seg_df1 = analyze_segments(time_filtered1, signal_filtered1, segment_duration, fixed_fundamental)
        seg_df2 = analyze_segments(time_filtered2, signal_filtered2, segment_duration, fixed_fundamental)

        st.write("#### Segments Signal 1")
        st.dataframe(seg_df1)
        st.write("#### Segments Signal 2")
        st.dataframe(seg_df2)

        # Visualisation avec meilleurs et pires segments
        fig3, ax3 = plt.subplots(2,1, figsize=(12,8))
        ax3[0].plot(time_filtered1, signal_filtered1, label="Signal 1")
        best1, worst1 = seg_df1["Score global"].idxmax(), seg_df1["Score global"].idxmin()
        ax3[0].axvspan(seg_df1.loc[best1,"t_start"], seg_df1.loc[best1,"t_end"], color="green", alpha=0.3, label="Meilleur segment")
        ax3[0].axvspan(seg_df1.loc[worst1,"t_start"], seg_df1.loc[worst1,"t_end"], color="red", alpha=0.3, label="Pire segment")
        ax3[0].legend()
        ax3[0].set_title("Signal 1 avec meilleurs/pire segments")

        ax3[1].plot(time_filtered2, signal_filtered2, color="orange", label="Signal 2")
        best2, worst2 = seg_df2["Score global"].idxmax(), seg_df2["Score global"].idxmin()
        ax3[1].axvspan(seg_df2.loc[best2,"t_start"], seg_df2.loc[best2,"t_end"], color="green", alpha=0.3, label="Meilleur segment")
        ax3[1].axvspan(seg_df2.loc[worst2,"t_start"], seg_df2.loc[worst2,"t_end"], color="red", alpha=0.3, label="Pire segment")
        ax3[1].legend()
        ax3[1].set_title("Signal 2 avec meilleurs/pire segments")

        st.pyplot(fig3)

    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {e}")
else:
    st.info("Veuillez télécharger les deux fichiers CSV pour commencer l'analyse.")
