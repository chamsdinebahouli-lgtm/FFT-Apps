import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

st.title("Analyse FFT de deux signaux – moteur DC → mouvement linéaire")

# === Chargement fichiers ===
uploaded_file1 = st.file_uploader("Chargez le premier fichier CSV (Signal 1)", type=["csv"])
uploaded_file2 = st.file_uploader("Chargez le deuxième fichier CSV (Signal 2)", type=["csv"])

# === Fenêtre temporelle ===
colt1, colt2 = st.columns(2)
with colt1:
    start_threshold = st.number_input("Exclure les N premières secondes", min_value=0.0, value=30.0, step=1.0)
with colt2:
    end_threshold = st.number_input("Exclure les N dernières secondes", min_value=0.0, value=20.0, step=1.0)

# === Paramètres électromécaniques ===
st.subheader("Paramètres électromécaniques")
colp1, colp2, colp3, colp4 = st.columns(4)
with colp1:
    screw_pitch_mm = st.number_input("Pas de vis (mm/tr)", min_value=0.001, value=5.0, step=0.1)
with colp2:
    gear_ratio = st.number_input("Rapport réduction (tr moteur / tr vis)", min_value=0.001, value=1.0, step=0.1)
with colp3:
    comm_events_per_rev = st.number_input("Événements de commutation par tour (k_comm)", min_value=1, value=6, step=1,
                                          help="Nombre de pics de commutation par tour moteur (collecteur/balayage).")
with colp4:
    pwm_freq_hint = st.number_input("Fréquence PWM (Hz, optionnel)", min_value=0.0, value=0.0, step=10.0,
                                    help="0 si inconnu. Sert au marquage du spectre et au calcul du ripple PWM.")

# === Variables / structures ===
def empty_state():
    return {
        "time": np.array([]), "signal": np.array([]),
        "time_f": np.array([]), "sig_f": np.array([]),
        "freqs": np.array([]), "mag": np.array([]),
        "f0": 0.0, "noise_p": 0.0, "SNR": 0.0, "THD": 0.0,
        "prominent": [], "rot_guess": 0.0, "rpm": 0.0, "v_mm_s": 0.0,
        "pwm_ripple_db": None,
        "top_peaks": []  # pour sélection manuelle
    }

S1, S2 = empty_state(), empty_state()
comparison_result = "Analyse non réalisée."

# === Fonctions utilitaires FFT / mesures ===
def analyze_signal(time, signal):
    if len(time) < 2:
        return None
    dt = time[1] - time[0]
    sig_c = signal - np.mean(signal)
    fft_vals = np.fft.fft(sig_c)
    freqs = np.fft.fftfreq(len(sig_c), d=dt)
    mask = freqs >= 0
    freqs_pos = freqs[mask]
    mag_pos = np.abs(fft_vals[mask]) / len(sig_c)

    # fondamentale = plus grand pic hors DC
    f0 = 0.0
    prominent = []
    if len(mag_pos) > 1:
        idx0 = np.argmax(mag_pos[1:]) + 1
        f0 = freqs_pos[idx0]
        prominent.append((f0, mag_pos[idx0]))

        # 5 pics supplémentaires
        sorted_idx = np.argsort(mag_pos[1:])[::-1] + 1
        for i in sorted_idx:
            f, m = freqs_pos[i], mag_pos[i]
            if abs(f - f0) > 1e-9:
                prominent.append((f, m))
            if len(prominent) >= 6:
                break

    # bruit 0-10 Hz hors fondamentale
    noise_p = 0.0
    for f, m in zip(freqs_pos, mag_pos):
        if 0 <= f <= 10 and abs(f - f0) > 1e-9:
            noise_p += m**2

    # SNR / THD (simple, sur pics retenus)
    if f0 > 0:
        fi = np.argmin(np.abs(freqs_pos - f0))
        p_fund = mag_pos[fi]**2
        p_harm = sum([m**2 for (f, m) in prominent if abs(f - f0) > 1e-9])
        SNR = 10*np.log10(p_fund / noise_p) if noise_p > 0 else np.inf
        THD = 10*np.log10(p_harm / p_fund) if p_fund > 0 else -np.inf
    else:
        SNR, THD = 0.0, 0.0

    return freqs_pos, mag_pos, f0, prominent, noise_p, SNR, THD

def extract_harmonics_table(prominent_list):
    if not prominent_list:
        return pd.DataFrame(columns=["k", "Fréquence (Hz)", "Amplitude", "Relatif (dB)"])
    amps = np.array([a for _, a in prominent_list])
    a_ref = amps.max() if amps.size else 1.0
    rows = []
    for i, (f, a) in enumerate(prominent_list):
        rel_db = 20*np.log10(a/a_ref) if a > 0 else -np.inf
        rows.append({"k": i+1, "Fréquence (Hz)": f, "Amplitude": a, "Relatif (dB)": rel_db})
    return pd.DataFrame(rows)

def guess_rotation_peak(freqs, mag, pwm_hint=0.0):
    if freqs.size == 0:
        return 0.0
    mask = (freqs > 0) & (freqs < 200.0)
    if pwm_hint > 0:
        avoid = (freqs > 0.95*pwm_hint) & (freqs < 1.05*pwm_hint)
        mask = mask & (~avoid)
    if not np.any(mask):
        return 0.0
    sub_mag = mag[mask]
    sub_freqs = freqs[mask]
    idx = np.argmax(sub_mag)
    return float(sub_freqs[idx])

def pwm_ripple_db(freqs, mag, pwm_f):
    if pwm_f <= 0 or freqs.size == 0:
        return None
    i_pwm = np.argmin(np.abs(freqs - pwm_f))
    a_pwm = mag[i_pwm]
    a_max = mag.max() if mag.size else 1.0
    return 20*np.log10(a_pwm / a_max) if a_pwm > 0 else -np.inf

def find_top_peaks(freqs, mag, n_peaks=8, fmin=0.2, fmax=200.0, pwm_hint=0.0, pwm_margin=0.05):
    """
    Retourne les N plus grands pics (freq, amp) dans [fmin, fmax],
    en excluant ±pwm_margin autour de la PWM si fournie.
    """
    if freqs.size == 0:
        return []
    mask = (freqs >= fmin) & (freqs <= fmax)
    if pwm_hint > 0:
        mask = mask & ~((freqs >= (1-pwm_margin)*pwm_hint) & (freqs <= (1+pwm_margin)*pwm_hint))
    f = freqs[mask]; a = mag[mask]
    if f.size == 0:
        return []
    idx_sorted = np.argsort(a)[::-1]  # décroissant
    peaks = []
    for i in idx_sorted[:n_peaks]:
        peaks.append((float(f[i]), float(a[i])))
    return peaks

def fill_from_df(df, state):
    if df is None or ('Time' not in df.columns) or ('Signal' not in df.columns):
        return state
    t, x = df['Time'].values, df['Signal'].values
    if t.size < 2:
        return state
    # fenêtre temporelle
    t_start = start_threshold
    t_end = t[-1] - end_threshold
    s_idx = np.argmax(t >= t_start)
    e_idx = len(t) - 1 - np.argmax(t[::-1] <= t_end)
    s_idx = int(s_idx); e_idx = int(e_idx)
    if e_idx < s_idx:
        return state
    state["time"], state["signal"] = t, x
    state["time_f"], state["sig_f"] = t[s_idx:e_idx+1], x[s_idx:e_idx+1]
    res = analyze_signal(state["time_f"], state["sig_f"])
    if res:
        freqs, mag, f0, prominent, noise_p, SNR, THD = res
        state["freqs"], state["mag"] = freqs, mag
        state["f0"], state["prominent"] = f0, prominent
        state["noise_p"], state["SNR"], state["THD"] = noise_p, SNR, THD
        # pics principaux pour sélection manuelle
        state["top_peaks"] = find_top_peaks(freqs, mag, n_peaks=12, fmin=0.2, fmax=200.0,
                                            pwm_hint=pwm_freq_hint if pwm_freq_hint > 0 else 0.0)
        # heuristique par défaut
        state["rot_guess"] = guess_rotation_peak(freqs, mag, pwm_freq_hint if pwm_freq_hint > 0 else 0.0)
        # PWM ripple
        state["pwm_ripple_db"] = pwm_ripple_db(freqs, mag, pwm_freq_hint) if pwm_freq_hint > 0 else None
        # RPM / vitesse linéaire (sera peut-être mis à jour si sélection manuelle)
        if state["rot_guess"] > 0 and comm_events_per_rev > 0:
            f_rot = state["rot_guess"] / comm_events_per_rev  # Hz (tr/s moteur)
            rpm = f_rot * 60.0
            f_vis = f_rot / gear_ratio  # tr/s vis
            v_mm_s = f_vis * screw_pitch_mm  # mm/s
            state["rpm"] = rpm
            state["v_mm_s"] = v_mm_s
    return state

# === Lecture CSV et analyse ===
if uploaded_file1 is not None and uploaded_file2 is not None:
    try:
        df1 = pd.read_csv(uploaded_file1, decimal=',')
        df2 = pd.read_csv(uploaded_file2, decimal=',')

        st.success("Les deux fichiers ont été chargés avec succès !")
        colh1, colh2 = st.columns(2)
        with colh1:
            st.subheader("Aperçu – Fichier 1")
            st.dataframe(df1.head())
        with colh2:
            st.subheader("Aperçu – Fichier 2")
            st.dataframe(df2.head())

        # Analyse
        S1 = fill_from_df(df1, S1)
        S2 = fill_from_df(df2, S2)

        # === Sélection manuelle du pic de commutation ===
        st.subheader("Sélection du pic de commutation (optionnelle)")
        colsel1, colsel2 = st.columns(2)

        with colsel1:
            st.markdown("**Signal 1**")
            manual1 = st.checkbox("Sélection manuelle (Signal 1)")
            if manual1 and len(S1["top_peaks"]) > 0:
                n1 = st.number_input("N pics affichés (S1)", min_value=1, max_value=len(S1["top_peaks"]), value=min(8, len(S1["top_peaks"])))
                options1 = S1["top_peaks"][:n1]
                labels1 = [f"{i+1}. {f:.3f} Hz (amp={a:.3g})" for i, (f, a) in enumerate(options1)]
                choice1 = st.selectbox("Choisir le pic (S1)", options=list(range(n1)), format_func=lambda i: labels1[i])
                chosen_f1 = options1[choice1][0]
                # mise à jour rot_guess / RPM / vitesse linéaire
                S1["rot_guess"] = chosen_f1
                if S1["rot_guess"] > 0 and comm_events_per_rev > 0:
                    f_rot = S1["rot_guess"] / comm_events_per_rev
                    S1["rpm"] = f_rot * 60.0
                    S1["v_mm_s"] = (f_rot / gear_ratio) * screw_pitch_mm

                # petit tableau des pics S1
                df_peaks1 = pd.DataFrame({"Rang": np.arange(1, n1+1),
                                          "Fréquence (Hz)": [f for f, _ in options1],
                                          "Amplitude": [a for _, a in options1]})
                st.dataframe(df_peaks1.style.format({"Fréquence (Hz)": "{:.4f}", "Amplitude": "{:.6f}"}))

        with colsel2:
            st.markdown("**Signal 2**")
            manual2 = st.checkbox("Sélection manuelle (Signal 2)")
            if manual2 and len(S2["top_peaks"]) > 0:
                n2 = st.number_input("N pics affichés (S2)", min_value=1, max_value=len(S2["top_peaks"]), value=min(8, len(S2["top_peaks"])))
                options2 = S2["top_peaks"][:n2]
                labels2 = [f"{i+1}. {f:.3f} Hz (amp={a:.3g})" for i, (f, a) in enumerate(options2)]
                choice2 = st.selectbox("Choisir le pic (S2)", options=list(range(n2)), format_func=lambda i: labels2[i])
                chosen_f2 = options2[choice2][0]
                S2["rot_guess"] = chosen_f2
                if S2["rot_guess"] > 0 and comm_events_per_rev > 0:
                    f_rot = S2["rot_guess"] / comm_events_per_rev
                    S2["rpm"] = f_rot * 60.0
                    S2["v_mm_s"] = (f_rot / gear_ratio) * screw_pitch_mm

                df_peaks2 = pd.DataFrame({"Rang": np.arange(1, n2+1),
                                          "Fréquence (Hz)": [f for f, _ in options2],
                                          "Amplitude": [a for _, a in options2]})
                st.dataframe(df_peaks2.style.format({"Fréquence (Hz)": "{:.4f}", "Amplitude": "{:.6f}"}))

        # === Graphiques ===
        st.subheader("Graphiques (temps et spectre)")
        if (S1["time_f"].size > 1) or (S2["time_f"].size > 1):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Signal 1 - temps
            if S1["time_f"].size > 1:
                axes[0,0].plot(S1["time_f"], S1["sig_f"])
                axes[0,0].set_title("Signal 1 (temps)")
                axes[0,0].set_xlabel("Temps (s)")
                axes[0,0].set_ylabel("Amplitude")
                axes[0,0].grid(True)
            else:
                axes[0,0].set_title("Signal 1 (temps) – N/A")

            # Signal 1 - spectre
            if S1["freqs"].size > 0:
                axes[0,1].stem(S1["freqs"], S1["mag"], basefmt=" ")
                axes[0,1].set_xlim(0, max(10, min(200, S1["freqs"].max())))
                axes[0,1].set_title("FFT – Signal 1")
                axes[0,1].set_xlabel("Fréquence (Hz)")
                axes[0,1].set_ylabel("Amplitude")
                axes[0,1].grid(True)
                # marquage PWM et pic sélectionné
                if pwm_freq_hint > 0:
                    axes[0,1].axvline(pwm_freq_hint, linestyle="--")
                if S1["rot_guess"] > 0:
                    axes[0,1].axvline(S1["rot_guess"], linestyle=":")
            else:
                axes[0,1].set_title("FFT – Signal 1 – N/A")

            # Signal 2 - temps
            if S2["time_f"].size > 1:
                axes[1,0].plot(S2["time_f"], S2["sig_f"], color='orange')
                axes[1,0].set_title("Signal 2 (temps)")
                axes[1,0].set_xlabel("Temps (s)")
                axes[1,0].set_ylabel("Amplitude")
                axes[1,0].grid(True)
            else:
                axes[1,0].set_title("Signal 2 (temps) – N/A")

            # Signal 2 - spectre
            if S2["freqs"].size > 0:
                axes[1,1].stem(S2["freqs"], S2["mag"], basefmt=" ", linefmt='orange')
                axes[1,1].set_xlim(0, max(10, min(200, S2["freqs"].max())))
                axes[1,1].set_title("FFT – Signal 2")
                axes[1,1].set_xlabel("Fréquence (Hz)")
                axes[1,1].set_ylabel("Amplitude")
                axes[1,1].grid(True)
                if pwm_freq_hint > 0:
                    axes[1,1].axvline(pwm_freq_hint, linestyle="--")
                if S2["rot_guess"] > 0:
                    axes[1,1].axvline(S2["rot_guess"], linestyle=":")
            else:
                axes[1,1].set_title("FFT – Signal 2 – N/A")

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Pas assez de points pour tracer les courbes.")

        # === Résultats numériques ===
        st.subheader("Indicateurs de qualité")
        colr1, colr2 = st.columns(2)
        with colr1:
            st.markdown("**Signal 1**")
            st.write(f"f₀ = {S1['f0']:.4f} Hz")
            st.write(f"SNR = {S1['SNR']:.2f} dB")
            st.write(f"THD = {S1['THD']:.2f} dB")
            st.write(f"Bruit (0–10 Hz, hors f₀) = {S1['noise_p']:.4f}")
            if S1["rot_guess"] > 0:
                st.write(f"Pic commutation ≈ {S1['rot_guess']:.3f} Hz → RPM ≈ {S1['rpm']:.1f} tr/min, Vitesse linéaire ≈ {S1['v_mm_s']:.2f} mm/s")
            if S1["pwm_ripple_db"] is not None:
                st.write(f"Ripple PWM (rel. au pic max) ≈ {S1['pwm_ripple_db']:.1f} dB")
        with colr2:
            st.markdown("**Signal 2**")
            st.write(f"f₀ = {S2['f0']:.4f} Hz")
            st.write(f"SNR = {S2['SNR']:.2f} dB")
            st.write(f"THD = {S2['THD']:.2f} dB")
            st.write(f"Bruit (0–10 Hz, hors f₀) = {S2['noise_p']:.4f}")
            if S2["rot_guess"] > 0:
                st.write(f"Pic commutation ≈ {S2['rot_guess']:.3f} Hz → RPM ≈ {S2['rpm']:.1f} tr/min, Vitesse linéaire ≈ {S2['v_mm_s']:.2f} mm/s")
            if S2["pwm_ripple_db"] is not None:
                st.write(f"Ripple PWM (rel. au pic max) ≈ {S2['pwm_ripple_db']:.1f} dB")

        # === Tables d'harmoniques (k, f, amplitude, relatif dB) ===
        st.subheader("Harmoniques")
        colh1, colh2 = st.columns(2)
        with colh1:
            st.write("**Signal 1**")
            df_h1 = extract_harmonics_table(S1["prominent"])
            st.dataframe(df_h1.style.format({"Fréquence (Hz)": "{:.4f}", "Amplitude": "{:.6f}", "Relatif (dB)": "{:.1f}"}))
        with colh2:
            st.write("**Signal 2**")
            df_h2 = extract_harmonics_table(S2["prominent"])
            st.dataframe(df_h2.style.format({"Fréquence (Hz)": "{:.4f}", "Amplitude": "{:.6f}", "Relatif (dB)": "{:.1f}"}))

        # === Comparaison simple ===
        if (S1["freqs"].size > 0) and (S2["freqs"].size > 0):
            if (S1["SNR"] > S2["SNR"]) and (S1["THD"] < S2["THD"]):
                comparison_result = "✅ Signal 1 globalement moins perturbé (SNR↑, THD↓)."
            elif (S2["SNR"] > S1["SNR"]) and (S2["THD"] < S1["THD"]):
                comparison_result = "✅ Signal 2 globalement moins perturbé (SNR↑, THD↓)."
            else:
                comparison_result = "⚖️ Compromis : SNR/THD en désaccord ou proches."
        st.subheader("Conclusion")
        st.write(comparison_result)

        # === Export CSV Résumé + Harmoniques ===
        st.subheader("Export")
        results_df = pd.DataFrame({
            "Signal": ["Signal 1", "Signal 2"],
            "f0_Hz": [S1["f0"], S2["f0"]],
            "SNR_dB": [S1["SNR"], S2["SNR"]],
            "THD_dB": [S1["THD"], S2["THD"]],
            "Bruit_0_10Hz": [S1["noise_p"], S2["noise_p"]],
            "Pic_commutation_Hz": [S1["rot_guess"], S2["rot_guess"]],
            "RPM": [S1["rpm"], S2["rpm"]],
            "V_lin_mm_s": [S1["v_mm_s"], S2["v_mm_s"]],
            "Ripple_PWM_dB": [S1["pwm_ripple_db"], S2["pwm_ripple_db"]]
        })

        csv_buf_res = io.StringIO()
        results_df.to_csv(csv_buf_res, index=False, sep=";")
        st.download_button("📥 Télécharger Résumé (CSV)", data=csv_buf_res.getvalue(),
                           file_name="resume_signaux.csv", mime="text/csv")

        # harmonics CSV
        if not S1["prominent"] == []:
            df_h1_csv = extract_harmonics_table(S1["prominent"])
            buf1 = io.StringIO(); df_h1_csv.to_csv(buf1, index=False, sep=";")
            st.download_button("📥 Harmoniques Signal 1 (CSV)", data=buf1.getvalue(),
                               file_name="harmoniques_signal1.csv", mime="text/csv")
        if not S2["prominent"] == []:
            df_h2_csv = extract_harmonics_table(S2["prominent"])
            buf2 = io.StringIO(); df_h2_csv.to_csv(buf2, index=False, sep=";")
            st.download_button("📥 Harmoniques Signal 2 (CSV)", data=buf2.getvalue(),
                               file_name="harmoniques_signal2.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {e}")
else:
    st.info("Veuillez télécharger les deux fichiers CSV pour commencer l'analyse.")
