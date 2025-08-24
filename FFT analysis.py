import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

st.title("Analyse FFT et comparaison au modèle continu")

# --- Sidebar paramètres ---
st.sidebar.title("⚙ Paramètres")
start_threshold = st.sidebar.number_input("Exclure N premières secondes", 0.0, 1000.0, 30.0)
end_threshold = st.sidebar.number_input("Exclure N dernières secondes", 0.0, 1000.0, 20.0)
fixed_fundamental = st.sidebar.number_input("Fréquence fondamentale forcée (Hz, 0=auto)", 0.0)
freq_display_max = st.sidebar.number_input("Fréquence max affichée sur la FFT (Hz)", 1.0, 100.0, 10.0)
uploaded_file1 = st.sidebar.file_uploader("CSV Signal 1", type="csv")
uploaded_file2 = st.sidebar.file_uploader("CSV Signal 2", type="csv")

# --- Définition FFT et indicateurs ---
def analyze_signal(time, signal, fixed_fundamental=0.0):
    dt = time[1] - time[0]
    sig_centered = signal - np.mean(signal)
    fft_vals = np.fft.fft(sig_centered)
    freqs = np.fft.fftfreq(len(sig_centered), d=dt)
    mask = freqs >= 0
    magnitude_pos = np.abs(fft_vals[mask]) / len(sig_centered)
    magnitude_pos[1:] *= 2
    freqs_pos = freqs[mask]
    
    # Détection fondamental et harmoniques
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
    noise_power = sum([m**2 for f,m in zip(freqs_pos, magnitude_pos) if 0<=f<=10 and all(abs(f-h[1])>tol for h in harmonics)])
    
    power_fund = harmonics[0][2]**2 if harmonics else 0
    power_harmo = sum([h[2]**2 for h in harmonics[1:]]) if harmonics else 0
    SNR = 10*np.log10(power_fund / noise_power) if noise_power>0 else np.inf
    THD = 10*np.log10(power_harmo / power_fund) if power_fund>0 else -np.inf
    amp_fundamental = harmonics[0][2] if harmonics else 0
    score_global = 0.4*SNR - 0.3*THD - 0.2*noise_power + 0.1*amp_fundamental
    return freqs_pos, magnitude_pos, fundamental_freq, harmonics, noise_power, SNR, THD, amp_fundamental, score_global

# --- Signal modèle et comparaison ---
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

# --- Traitement fichiers ---
if uploaded_file1 and uploaded_file2:
    try:
        df1 = pd.read_csv(uploaded_file1, decimal=',')
        df2 = pd.read_csv(uploaded_file2, decimal=',')
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
        signal_ideal1 = generate_ideal_signal(time_filtered1, signal_filtered1, 'mean')
        signal_ideal2 = generate_ideal_signal(time_filtered2, signal_filtered2, 'mean')
        diff1, rmse1, freqs_diff1, mag_diff1, noise_power_diff1 = compare_to_model(time_filtered1, signal_filtered1, signal_ideal1)
        diff2, rmse2, freqs_diff2, mag_diff2, noise_power_diff2 = compare_to_model(time_filtered2, signal_filtered2, signal_ideal2)

        # --- Visualisation Signal réel vs idéal ---
        fig1, axes1 = plt.subplots(2,1, figsize=(12,6))
        axes1[0].plot(time_filtered1, signal_filtered1, label='Réel 1')
        axes1[0].plot(time_filtered1, signal_ideal1, '--', label='Idéal 1')
        axes1[0].set_title(f"Signal 1 (RMSE={rmse1:.4f})"); axes1[0].legend()
        axes1[1].plot(time_filtered2, signal_filtered2, label='Réel 2', color='orange')
        axes1[1].plot(time_filtered2, signal_ideal2, '--', label='Idéal 2', color='green')
        axes1[1].set_title(f"Signal 2 (RMSE={rmse2:.4f})"); axes1[1].legend()
        st.pyplot(fig1)

        # --- FFT des signaux réels et des écarts ---
        fig3, axes3 = plt.subplots(2,2, figsize=(14,8))
        # Signal 1 réel
        axes3[0,0].stem(freqs_pos1, magnitude_pos1, linefmt='-', markerfmt='o', basefmt=" ")
        axes3[0,0].set_xlim(0,freq_display_max)
        axes3[0,0].set_title("FFT Signal 1 réel")
        # Signal 1 écart
        axes3[0,1].stem(freqs_diff1, mag_diff1, linefmt='-', markerfmt='o', basefmt=" ")
        axes3[0,1].set_xlim(0,freq_display_max)
        axes3[0,1].set_title("FFT Signal 1 écart vs modèle")
        # Signal 2 réel
        axes3[1,0].stem(freqs_pos2, magnitude_pos2, linefmt='-', markerfmt='o', basefmt=" ")
        axes3[1,0].set_xlim(0,freq_display_max)
        axes3[1,0].set_title("FFT Signal 2 réel")
        # Signal 2 écart
        axes3[1,1].stem(freqs_diff2, mag_diff2, linefmt='-', markerfmt='o', basefmt=" ")
        axes3[1,1].set_xlim(0,freq_display_max)
        axes3[1,1].set_title("FFT Signal 2 écart vs modèle")
        plt.tight_layout()
        st.pyplot(fig3)

        # --- Affichage indicateurs et score global ---
        st.write("### Indicateurs et Score")
        for i, (fund,SNRv,THDv,noise,harms,amp,score,rmse) in enumerate([
            (fundamental_frequency1,SNR1,THD1,noise_power1, harmonics1, amp_fund1, score_global1, rmse1),
            (fundamental_frequency2,SNR2,THD2,noise_power2, harmonics2, amp_fund2, score_global2, rmse2)
        ], start=1):
            st.write(f"**Signal {i}** :")
            st.write(f"Fréquence fondamentale = {fund:.4f} Hz")
            st.write(f"Amplitude fondamentale = {amp:.4f}")
            st.write(f"SNR = {'∞' if SNRv==np.inf else f'{SNRv:.2f} dB'}")
            st.write(f"THD = {'-∞' if THDv==-np.inf else f'{THDv:.2f} dB'}")
            st.write(f"Bruit (0-10Hz hors harmoniques) = {noise:.4f}")
            st.write(f"RMSE vs modèle = {rmse:.4f}")
            st.write(f"Score global = {score:.2f}")
            st.write("Harmoniques :")
            st.dataframe(pd.DataFrame(harms, columns=["Ordre","Fréquence (Hz)","Amplitude"]))

        # --- Comparaison globale ---
       # --- Comparaison globale détaillée avec mise en couleur ---

# Construction du tableau comparatif
comparison_data = {
    "Critère": ["RMSE vs modèle", "SNR (dB)", "THD (dB)", "Puissance de bruit", "Score global"],
    "Signal 1": [rmse1, SNR1, THD1, noise_power1, score_global1],
    "Signal 2": [rmse2, SNR2, THD2, noise_power2, score_global2],
}

comparison_df = pd.DataFrame(comparison_data)

# Fonction pour colorer les meilleures valeurs
def highlight_best(val1, val2, crit):
    if crit == "RMSE vs modèle" or crit == "THD (dB)" or crit == "Puissance de bruit":
        # Plus petit = meilleur
        if val1 < val2:
            return ["background-color: lightgreen", "background-color: salmon"]
        elif val2 < val1:
            return ["background-color: salmon", "background-color: lightgreen"]
        else:
            return ["background-color: lightyellow", "background-color: lightyellow"]
    else:
        # Plus grand = meilleur
        if val1 > val2:
            return ["background-color: lightgreen", "background-color: salmon"]
        elif val2 > val1:
            return ["background-color: salmon", "background-color: lightgreen"]
        else:
            return ["background-color: lightyellow", "background-color: lightyellow"]

# Application du style ligne par ligne
styled_df = comparison_df.style.apply(
    lambda row: highlight_best(row["Signal 1"], row["Signal 2"], row["Critère"]),
    axis=1, subset=["Signal 1", "Signal 2"]
)

st.write("### Comparaison globale")
st.dataframe(styled_df, use_container_width=True)

# --- Analyse textuelle automatique (comme avant) ---
analysis_comments = []

# RMSE
if rmse1 < rmse2:
    analysis_comments.append("🔹 **Signal 1** suit mieux le modèle idéal (RMSE plus faible).")
elif rmse2 < rmse1:
    analysis_comments.append("🔹 **Signal 2** suit mieux le modèle idéal (RMSE plus faible).")
else:
    analysis_comments.append("🔹 Les deux signaux suivent le modèle idéal de façon similaire (RMSE équivalent).")

# SNR
if SNR1 > SNR2:
    analysis_comments.append("🔹 **Signal 1** est moins bruité (SNR supérieur).")
elif SNR2 > SNR1:
    analysis_comments.append("🔹 **Signal 2** est moins bruité (SNR supérieur).")
else:
    analysis_comments.append("🔹 Les deux signaux présentent un niveau de bruit similaire (SNR équivalent).")

# THD
if THD1 < THD2:
    analysis_comments.append("🔹 **Signal 1** présente une distorsion harmonique plus faible → meilleure pureté spectrale.")
elif THD2 < THD1:
    analysis_comments.append("🔹 **Signal 2** présente une distorsion harmonique plus faible → meilleure pureté spectrale.")
else:
    analysis_comments.append("🔹 Les deux signaux ont une distorsion harmonique similaire.")

# Bruit
if noise_power1 < noise_power2:
    analysis_comments.append("🔹 **Signal 1** contient moins de bruit résiduel dans la bande 0-10 Hz.")
elif noise_power2 < noise_power1:
    analysis_comments.append("🔹 **Signal 2** contient moins de bruit résiduel dans la bande 0-10 Hz.")
else:
    analysis_comments.append("🔹 Les deux signaux ont un bruit comparable dans la bande 0-10 Hz.")

# Score global
if score_global1 > score_global2:
    final_verdict = "✅ **Signal 1 est globalement meilleur selon le score combiné.**"
elif score_global2 > score_global1:
    final_verdict = "✅ **Signal 2 est globalement meilleur selon le score combiné.**"
else:
    final_verdict = "✅ **Les deux signaux sont équivalents selon le score combiné.**"

# Affichage de l’analyse
st.markdown("#### Analyse des résultats")
for comment in analysis_comments:
    st.markdown(comment)

st.markdown(f"### Verdict final\n{final_verdict}")

    except Exception as e:
        st.error(f"Erreur : {e}")
else:
    st.info("Veuillez télécharger les deux fichiers CSV pour commencer l'analyse.")
