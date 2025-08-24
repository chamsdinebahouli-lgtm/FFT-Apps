import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- Fonction d'analyse FFT ---
def analyze_signal(time, signal, fixed_freq=0.0, n_harmonics=5):
    N = len(signal)
    if N == 0:
        return None, None, 0, 0, 0, 0, 0, [], 0

    T = (time[-1] - time[0]) / N
    Fs = 1.0 / T
    freqs = np.fft.rfftfreq(N, d=1/Fs)
    fft_vals = np.fft.rfft(signal) / N
    mag = 2 * np.abs(fft_vals)

    # Détection de la fondamentale
    if fixed_freq > 0:
        f0 = fixed_freq
        amp0 = np.interp(f0, freqs, mag)
    else:
        idx_max = np.argmax(mag[1:]) + 1
        f0 = freqs[idx_max]
        amp0 = mag[idx_max]

    # Harmoniques
    harmonics = []
    for k in range(2, n_harmonics+1):
        fk = k * f0
        if fk < freqs[-1]:
            ak = np.interp(fk, freqs, mag)
            harmonics.append((k, fk, ak))

    # THD
    thd = np.sqrt(np.sum([h[2]**2 for h in harmonics])) / amp0 if amp0 > 0 else 0

    # Bruit (reste hors fondamentale et harmoniques)
    harm_freqs = [h[1] for h in harmonics] + [f0]
    noise_mask = np.ones_like(freqs, dtype=bool)
    for f in harm_freqs:
        noise_mask &= np.abs(freqs - f) > 1.0
    noise_power = np.mean(mag[noise_mask]) if np.any(noise_mask) else 0

    # SNR
    snr = (amp0 / noise_power) if noise_power > 0 else np.inf

    # Score global
    score = (snr / (1 + thd + noise_power)) * amp0

    return freqs, mag, f0, amp0, snr, thd, noise_power, harmonics, score


# --- Application Streamlit ---
st.title("🔎 Analyse spectrale de deux signaux (FFT)")

# Paramètres utilisateur
st.sidebar.header("⚙️ Paramètres")
fixed_freq = st.sidebar.number_input("Forcer la fréquence fondamentale (Hz, 0 = auto)", value=0.0)
t_start_exclude = st.sidebar.number_input("Exclure début (secondes)", value=0.0, min_value=0.0)
t_end_exclude = st.sidebar.number_input("Exclure fin (secondes)", value=0.0, min_value=0.0)

# Import des signaux
uploaded_file1 = st.file_uploader("Charger le fichier CSV du Signal 1", type=["csv"])
uploaded_file2 = st.file_uploader("Charger le fichier CSV du Signal 2", type=["csv"])

if uploaded_file1 and uploaded_file2:
    data1 = pd.read_csv(uploaded_file1)
    data2 = pd.read_csv(uploaded_file2)

    time1, signal1 = data1.iloc[:, 0].values, data1.iloc[:, 1].values
    time2, signal2 = data2.iloc[:, 0].values, data2.iloc[:, 1].values

    # --- Exclusion des zones temporelles ---
    def cut_signal(time, signal, t_start_exclude, t_end_exclude):
        mask = (time >= time[0] + t_start_exclude) & (time <= time[-1] - t_end_exclude)
        return time[mask], signal[mask]

    time1, signal1 = cut_signal(time1, signal1, t_start_exclude, t_end_exclude)
    time2, signal2 = cut_signal(time2, signal2, t_start_exclude, t_end_exclude)

    # Vérif sécurité si signal vide
    if len(time1) == 0 or len(time2) == 0:
        st.error("⚠️ Après exclusion du début et de la fin, un des signaux est vide. Réduisez les valeurs d’exclusion.")
    else:
        # --- Affichage interactif Plotly ---
        st.subheader("📉 Signaux après exclusion temporelle (interactif)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time1, y=signal1, mode="lines", name="Signal 1", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=time2, y=signal2, mode="lines", name="Signal 2", line=dict(color="red")))
        fig.update_layout(
            title="Signaux tronqués",
            xaxis_title="Temps (s)",
            yaxis_title="Amplitude",
            hovermode="x unified",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Analyse FFT ---
        freqs1, mag1, f0_1, amp1, SNR1, THD1, noise1, harms1, score1 = analyze_signal(time1, signal1, fixed_freq)
        freqs2, mag2, f0_2, amp2, SNR2, THD2, noise2, harms2, score2 = analyze_signal(time2, signal2, fixed_freq)

        # --- Résultats numériques ---
        st.subheader("📊 Paramètres des deux signaux")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Signal 1 :**")
            st.write(f"Fréquence fondamentale = {f0_1:.2f} Hz")
            st.write(f"Amplitude fondamentale = {amp1:.4f}")
            st.write(f"SNR = {SNR1:.2f}")
            st.write(f"THD = {THD1:.4f}")
            st.write(f"Bruit = {noise1:.4f}")
            st.write(f"Score global = {score1:.2f}")

        with col2:
            st.markdown(f"**Signal 2 :**")
            st.write(f"Fréquence fondamentale = {f0_2:.2f} Hz")
            st.write(f"Amplitude fondamentale = {amp2:.4f}")
            st.write(f"SNR = {SNR2:.2f}")
            st.write(f"THD = {THD2:.4f}")
            st.write(f"Bruit = {noise2:.4f}")
            st.write(f"Score global = {score2:.2f}")

        # --- Comparaison globale ---
        st.subheader("⚖️ Comparaison globale")
        if score1 > score2:
            st.success("✅ Le Signal 1 est globalement de meilleure qualité.")
        elif score2 > score1:
            st.success("✅ Le Signal 2 est globalement de meilleure qualité.")
        else:
            st.info("⚖️ Les deux signaux sont de qualité équivalente.")

        st.markdown("""
        **Interprétation :**
        - Un **SNR élevé** = meilleure clarté du signal (peu de bruit).
        - Un **THD faible** = faible distorsion harmonique → signal plus pur.
        - Un **bruit faible** = stabilité du système.
        - Une **amplitude fondamentale élevée** = puissance utile plus importante.
        - Le **score global** combine ces critères pour une comparaison synthétique.
        """)
