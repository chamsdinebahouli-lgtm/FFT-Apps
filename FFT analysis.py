import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Fonction d'analyse FFT ---
def analyze_signal(time, signal, fixed_fundamental=0):
    N = len(signal)
    T = (time[-1] - time[0]) / N
    fs = 1.0 / (time[1] - time[0])  # fréquence d’échantillonnage

    # FFT
    fft_values = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1 / fs)
    magnitude = np.abs(fft_values) / N
    magnitude = 2 * magnitude[:N // 2]
    freqs_pos = freqs[:N // 2]

    # Détection fondamentale
    if fixed_fundamental > 0:
        fundamental_freq = fixed_fundamental
        magnitude_interp = np.interp(fundamental_freq, freqs_pos, magnitude)
        harmonics = [(1, fundamental_freq, magnitude_interp)]
        for n in range(2, 11):
            target = n * fundamental_freq
            if target <= freqs_pos[-1]:
                magnitude_interp = np.interp(target, freqs_pos, magnitude)
                harmonics.append((n, target, magnitude_interp))
    else:
        idx_max = np.argmax(magnitude[1:]) + 1
        fundamental_freq = freqs_pos[idx_max]
        harmonics = [(1, fundamental_freq, magnitude[idx_max])]
        for n in range(2, 11):
            target = n * fundamental_freq
            if target <= freqs_pos[-1]:
                idx = np.argmin(np.abs(freqs_pos - target))
                harmonics.append((n, freqs_pos[idx], magnitude[idx]))

    # Paramètres
    amp_fund = harmonics[0][2]
    harm_amps = np.array([h[2] for h in harmonics[1:]])
    harm_power = np.sum(harm_amps ** 2)
    noise_power = np.sum(magnitude ** 2) - (amp_fund ** 2 + harm_power)
    SNR = 10 * np.log10(amp_fund ** 2 / noise_power) if noise_power > 0 else np.inf
    THD = 10 * np.log10(harm_power / amp_fund ** 2) if amp_fund > 0 else -np.inf
    score_global = SNR - THD - 10 * noise_power

    return freqs_pos, magnitude, fundamental_freq, amp_fund, SNR, THD, noise_power, harmonics, score_global

# --- Application Streamlit ---
st.title("🔎 Analyse spectrale de deux signaux (FFT)")

# Paramètres utilisateur
st.sidebar.header("⚙️ Paramètres")
fixed_freq = st.sidebar.number_input("Forcer la fréquence fondamentale (Hz, 0 = auto)", value=0.0)

# Import des signaux
uploaded_file1 = st.file_uploader("Charger le fichier CSV du Signal 1", type=["csv"])
uploaded_file2 = st.file_uploader("Charger le fichier CSV du Signal 2", type=["csv"])

if uploaded_file1 and uploaded_file2:
    data1 = pd.read_csv(uploaded_file1)
    data2 = pd.read_csv(uploaded_file2)

    time1, signal1 = data1.iloc[:, 0].values, data1.iloc[:, 1].values
    time2, signal2 = data2.iloc[:, 0].values, data2.iloc[:, 1].values

    # Analyse FFT
    freqs1, mag1, f0_1, amp1, SNR1, THD1, noise1, harms1, score1 = analyze_signal(time1, signal1, fixed_freq)
    freqs2, mag2, f0_2, amp2, SNR2, THD2, noise2, harms2, score2 = analyze_signal(time2, signal2, fixed_freq)

    # --- Graphiques temporels ---
    st.subheader("⏱️ Signaux temporels")
    fig, ax = plt.subplots()
    ax.plot(time1, signal1, label="Signal 1")
    ax.plot(time2, signal2, label="Signal 2")
    ax.legend()
    ax.set_xlabel("Temps [s]")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # --- Spectres FFT ---
    st.subheader("📊 Spectres de fréquences (FFT)")
    fig, ax = plt.subplots()
    ax.plot(freqs1, mag1, label="Signal 1")
    ax.plot(freqs2, mag2, label="Signal 2")
    ax.set_xlim(0, max(f0_1, f0_2) * 10)
    ax.legend()
    ax.set_xlabel("Fréquence [Hz]")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # --- Tableau des harmoniques ---
    st.subheader("📋 Harmoniques détectées")
    df_harmonics = pd.DataFrame({
        "Ordre": [h[0] for h in harms1],
        "Fréquence Signal 1 [Hz]": [h[1] for h in harms1],
        "Amplitude Signal 1": [h[2] for h in harms1],
        "Fréquence Signal 2 [Hz]": [h[1] for h in harms2],
        "Amplitude Signal 2": [h[2] for h in harms2],
    })
    st.dataframe(df_harmonics)

    # --- Résultats comparatifs globaux ---
    st.header("🔎 Comparaison globale")

    st.write("### Paramètres principaux")
    st.write(f"**Signal 1 :** f₀ = {f0_1:.4f} Hz, "
             f"Amplitude = {amp1:.4f}, SNR = {SNR1:.2f} dB, "
             f"THD = {THD1:.2f} dB, Bruit = {noise1:.4e}, Score global = {score1:.2f}")
    st.write(f"**Signal 2 :** f₀ = {f0_2:.4f} Hz, "
             f"Amplitude = {amp2:.4f}, SNR = {SNR2:.2f} dB, "
             f"THD = {THD2:.2f} dB, Bruit = {noise2:.4e}, Score global = {score2:.2f}")

    if score1 > score2:
        st.success("✅ **Signal 1 est globalement meilleur** selon les critères combinés.")
    elif score2 > score1:
        st.success("✅ **Signal 2 est globalement meilleur** selon les critères combinés.")
    else:
        st.info("⚖️ Les deux signaux présentent une qualité très proche.")

    # --- Commentaires qualitatifs ---
    st.write("### Commentaires qualitatifs complémentaires")

    # Stabilité de la fréquence fondamentale
    if abs(f0_1 - f0_2) < 0.1:
        st.write("⚙️ Les deux signaux ont une fréquence fondamentale très proche → bonne stabilité du moteur.")
    else:
        st.write("⚠️ Différence notable entre les fréquences fondamentales → possible variation du régime moteur.")

    # Amplitude de la fondamentale
    if abs(amp1 - amp2) < 0.01:
        st.write("📊 Les amplitudes fondamentales sont similaires → la puissance utile reste comparable.")
    else:
        strongest = "Signal 1" if amp1 > amp2 else "Signal 2"
        st.write(f"📊 {strongest} a une amplitude fondamentale plus élevée → meilleure transmission d’énergie utile.")

    # Distorsion harmonique
    if THD1 < THD2:
        st.write("🎶 Signal 1 présente moins de distorsion harmonique → spectre plus propre.")
    elif THD2 < THD1:
        st.write("🎶 Signal 2 présente moins de distorsion harmonique → spectre plus propre.")
    else:
        st.write("🎶 Les deux signaux ont un niveau de distorsion harmonique similaire.")

    # Rapport signal/bruit
    if SNR1 > SNR2:
        st.write("🔊 Signal 1 possède un meilleur rapport signal/bruit → moins de perturbations parasites.")
    elif SNR2 > SNR1:
        st.write("🔊 Signal 2 possède un meilleur rapport signal/bruit → moins de perturbations parasites.")
    else:
        st.write("🔊 Les deux signaux présentent un rapport signal/bruit comparable.")

    # Niveau de bruit
    if noise1 < noise2:
        st.write("🌐 Signal 1 a un niveau de bruit global plus faible → meilleure qualité.")
    elif noise2 < noise1:
        st.write("🌐 Signal 2 a un niveau de bruit global plus faible → meilleure qualité.")
    else:
        st.write("🌐 Les deux signaux ont un niveau de bruit similaire.")
