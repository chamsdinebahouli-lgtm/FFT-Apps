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

# Initialize variables to avoid NameError if files are not uploaded or analysis fails
df1 = None
df2 = None
time_filtered1 = np.array([])
signal_filtered1 = np.array([])
fundamental_frequency1 = 0
prominent_freqs1 = []
freqs_pos1 = np.array([])
magnitude_pos1 = np.array([])
noise_power1 = 0

time_filtered2 = np.array([])
signal_filtered2 = np.array([])
fundamental_frequency2 = 0
prominent_freqs2 = []
freqs_pos2 = np.array([])
magnitude_pos2 = np.array([])
noise_power2 = 0

comparison_result = "Aucune comparaison n'a pu être effectuée."


if uploaded_file1 is not None and uploaded_file2 is not None:
    try:
        # Read the uploaded CSV files into pandas DataFrames
        # Assuming the CSV uses comma as decimal separator, similar to the original notebook
        df1 = pd.read_csv(uploaded_file1, decimal=',')
        df2 = pd.read_csv(uploaded_file2, decimal=',')

        # Display a success message
        st.success("Les deux fichiers ont été chargés avec succès !")

        # Optionally display the head of the DataFrames for confirmation
        st.subheader("Aperçu des données - Fichier 1")
        st.dataframe(df1.head())

        st.subheader("Aperçu des données - Fichier 2")
        st.dataframe(df2.head())

        # --- Process Signal 1 ---
        if 'Time' in df1.columns and 'Signal' in df1.columns:
            time1 = df1['Time'].values
            signal1 = df1['Signal'].values

            # Apply time slicing for Signal 1
            time_threshold_start1 = start_threshold
            time_threshold_end_abs1 = time1[-1] - end_threshold

            start_index1 = np.argmax(time1 >= time_threshold_start1)
            end_index_candidate1 = len(time1) - 1 - np.argmax(time1[::-1] <= time_threshold_end_abs1)

            if 0 <= end_index_candidate1 < len(time1):
                 end_index1 = end_index_candidate1
            else:
                 end_index1 = len(time1) - 1
                 if end_index1 < start_index1:
                      start_index1 = end_index1


            if end_index1 >= start_index1:
                time_filtered1 = time1[start_index1:end_index1+1]
                signal_filtered1 = signal1[start_index1:end_index1+1]
                st.write(f"Analyse des données du Signal 1 de {time_filtered1[0]:.2f} à {time_filtered1[-1]:.2f} secondes.")

                if len(time_filtered1) > 1:
                    # FFT for Signal 1
                    dt1 = time_filtered1[1] - time_filtered1[0]
                    fs1 = 1 / dt1

                    signal1_centered = signal_filtered1 - np.mean(signal_filtered1)
                    fft_vals1 = np.fft.fft(signal1_centered)
                    freqs1 = np.fft.fftfreq(len(signal1_centered), d=dt1)

                    mask1 = freqs1 >= 0
                    freqs_pos1 = freqs1[mask1]
                    magnitude_pos1 = np.abs(fft_vals1[mask1]) / len(signal1_centered)

                    # === Find fundamental frequency and prominent harmonics for Signal 1 ===
                    fundamental_frequency1 = 0
                    prominent_freqs1 = []
                    if len(magnitude_pos1) > 1:
                        fundamental_freq_index1 = np.argmax(magnitude_pos1[1:]) + 1
                        fundamental_frequency1 = freqs_pos1[fundamental_freq_index1]
                        prominent_freqs1.append((fundamental_frequency1, magnitude_pos1[fundamental_freq_index1]))

                    sorted_indices1 = np.argsort(magnitude_pos1[1:])[::-1] + 1
                    num_harmonics_to_display = 5
                    displayed_harmonics_count1 = 0
                    for i in range(min(len(sorted_indices1), len(freqs_pos1) - 1)):
                         freq = freqs_pos1[sorted_indices1[i]]
                         mag = magnitude_pos1[sorted_indices1[i]]
                         if abs(freq - fundamental_frequency1) > 1e-9 and displayed_harmonics_count1 < num_harmonics_to_display:
                             prominent_freqs1.append((freq, mag))
                             displayed_harmonics_count1 += 1

                    # Calculate noise power for Signal 1
                    noise_freq_min = 0.0 # Define the frequency range for noise calculation
                    noise_freq_max = 10.0
                    frequency_tolerance = 1e-9 # Tolerance for excluding the fundamental frequency

                    noise_power1 = 0
                    for i in range(len(freqs_pos1)):
                        freq = freqs_pos1[i]
                        mag = magnitude_pos1[i]
                        # Include frequencies within the noise range, excluding the fundamental
                        if noise_freq_min <= freq <= noise_freq_max and abs(freq - fundamental_frequency1) > frequency_tolerance:
                            noise_power1 += mag**2 # Using magnitude squared for power

                else:
                    st.warning("Pas assez de points de données pour le Signal 1 après application des seuils temporels pour effectuer l'analyse FFT.")
                    fundamental_frequency1 = 0
                    prominent_freqs1 = []
                    freqs_pos1 = np.array([])
                    magnitude_pos1 = np.array([])
                    noise_power1 = 0


            else:
                st.warning("La plage temporelle spécifiée est invalide pour le Signal 1. Veuillez ajuster les seuils.")
                time_filtered1 = np.array([])
                signal_filtered1 = np.array([])
                fundamental_frequency1 = 0
                prominent_freqs1 = []
                freqs_pos1 = np.array([])
                magnitude_pos1 = np.array([])
                noise_power1 = 0

        else:
            st.error("Le fichier CSV du Signal 1 doit contenir les colonnes 'Time' et 'Signal'.")
            fundamental_frequency1 = 0
            prominent_freqs1 = []
            freqs_pos1 = np.array([])
            magnitude_pos1 = np.array([])
            time_filtered1 = np.array([])
            signal_filtered1 = np.array([])
            noise_power1 = 0


        # --- Process Signal 2 ---
        if 'Time' in df2.columns and 'Signal' in df2.columns:
            time2 = df2['Time'].values
            signal2 = df2['Signal'].values

            # Apply time slicing for Signal 2
            time_threshold_start2 = start_threshold
            time_threshold_end_abs2 = time2[-1] - end_threshold

            start_index2 = np.argmax(time2 >= time_threshold_start2)
            end_index_candidate2 = len(time2) - 1 - np.argmax(time2[::-1] <= time_threshold_end_abs2)

            if 0 <= end_index_candidate2 < len(time2):
                 end_index2 = end_index_candidate2
            else:
                 end_index2 = len(time2) - 1
                 if end_index2 < start_index2:
                      start_index2 = end_index2


            if end_index2 >= start_index2:
                time_filtered2 = time2[start_index2:end_index2+1]
                signal_filtered2 = signal2[start_index2:end_index2+1]
                st.write(f"Analyse des données du Signal 2 de {time_filtered2[0]:.2f} à {time_filtered2[-1]:.2f} secondes.")

                if len(time_filtered2) > 1:
                    # FFT for Signal 2
                    dt2 = time_filtered2[1] - time_filtered2[0]
                    fs2 = 1 / dt2

                    signal2_centered = signal_filtered2 - np.mean(signal_filtered2)
                    fft_vals2 = np.fft.fft(signal2_centered)
                    freqs2 = np.fft.fftfreq(len(signal2_centered), d=dt2)

                    mask2 = freqs2 >= 0
                    freqs_pos2 = freqs2[mask2]
                    magnitude_pos2 = np.abs(fft_vals2[mask2]) / len(signal2_centered)

                    # === Find fundamental frequency and prominent harmonics for Signal 2 ===
                    fundamental_frequency2 = 0
                    prominent_freqs2 = []
                    if len(magnitude_pos2) > 1:
                        fundamental_freq_index2 = np.argmax(magnitude_pos2[1:]) + 1
                        fundamental_frequency2 = freqs_pos2[fundamental_freq_index2]
                        prominent_freqs2.append((fundamental_frequency2, magnitude_pos2[fundamental_freq_index2]))

                    sorted_indices2 = np.argsort(magnitude_pos2[1:])[::-1] + 1
                    displayed_harmonics_count2 = 0
                    for i in range(min(len(sorted_indices2), len(freqs_pos2) - 1)):
                         freq = freqs_pos2[sorted_indices2[i]]
                         mag = magnitude_pos2[sorted_indices2[i]]
                         if abs(freq - fundamental_frequency2) > 1e-9 and displayed_harmonics_count2 < num_harmonics_to_display:
                             prominent_freqs2.append((freq, mag))
                             displayed_harmonics_count2 += 1

                    # Calculate noise power for Signal 2
                    noise_power2 = 0
                    for i in range(len(freqs_pos2)):
                        freq = freqs_pos2[i]
                        mag = magnitude_pos2[i]
                        # Include frequencies within the noise range, excluding the fundamental
                        if noise_freq_min <= freq <= noise_freq_max and abs(freq - fundamental_frequency2) > frequency_tolerance:
                            noise_power2 += mag**2 # Using magnitude squared for power

                else:
                    st.warning("Pas assez de points de données pour le Signal 2 après application des seuils temporels pour effectuer l'analyse FFT.")
                    fundamental_frequency2 = 0
                    prominent_freqs2 = []
                    freqs_pos2 = np.array([])
                    magnitude_pos2 = np.array([])
                    noise_power2 = 0

            else:
                st.warning("La plage temporelle spécifiée est invalide pour le Signal 2. Veuillez ajuster les seuils.")
                time_filtered2 = np.array([])
                signal_filtered2 = np.array([])
                fundamental_frequency2 = 0
                prominent_freqs2 = []
                freqs_pos2 = np.array([])
                magnitude_pos2 = np.array([])
                noise_power2 = 0

        else:
            st.error("Le fichier CSV du Signal 2 doit contenir les colonnes 'Time' et 'Signal'.")
            fundamental_frequency2 = 0
            prominent_freqs2 = []
            freqs_pos2 = np.array([])
            magnitude_pos2 = np.array([])
            time_filtered2 = np.array([])
            signal_filtered2 = np.array([])
            noise_power2 = 0


        # --- Compare Signals ---
        if (len(time_filtered1) > 1 and len(freqs_pos1) > 0) and (len(time_filtered2) > 1 and len(freqs_pos2) > 0):
            # Comparison based on fundamental frequency magnitude
            mag_fundamental1 = 0
            if fundamental_frequency1 != 0:
                fundamental_index1 = np.argmin(np.abs(freqs_pos1 - fundamental_frequency1))
                mag_fundamental1 = magnitude_pos1[fundamental_index1]

            mag_fundamental2 = 0
            if fundamental_frequency2 != 0:
                 fundamental_index2 = np.argmin(np.abs(freqs_pos2 - fundamental_frequency2))
                 mag_fundamental2 = magnitude_pos2[fundamental_index2]

            if mag_fundamental1 < mag_fundamental2:
                comparison_result = "Signal 1 est potentiellement meilleur (amplitude fondamentale plus basse)."
            elif mag_fundamental2 < mag_fundamental1:
                comparison_result = "Signal 2 est potentiellement meilleur (amplitude fondamentale plus basse)."
            else:
                # If fundamental frequencies are similar, compare based on noise power
                if noise_power1 < noise_power2:
                    comparison_result2 = "Signal 1 est potentiellement meilleur (moins de bruit)."
                elif noise_power2 < noise_power1:
                    comparison_result2 = "Signal 2 est potentiellement meilleur (moins de bruit)."
                else:
                    comparison_result2 = "Les signaux sont similaires selon les critères d'analyse."
        else:
            comparison_result = "Analyse FFT incomplète pour les deux signaux. Comparaison non possible."


        # --- Display Results ---
        st.subheader("Résultats de l'analyse")

        if (len(time_filtered1) > 1 and len(freqs_pos1) > 0) or (len(time_filtered2) > 1 and len(freqs_pos2) > 0):
            # === Tracés ===
            fig, axes = plt.subplots(2, 2, figsize=(12, 10)) # 2 rows for two signals, 2 columns for time/freq plots

            # Signal temporel 1
            if len(time_filtered1) > 1:
                axes[0, 0].plot(time_filtered1, signal_filtered1, label="Signal 1 filtré")
                axes[0, 0].set_xlabel("Temps (s)")
                axes[0, 0].set_ylabel("Amplitude")
                axes[0, 0].set_title("Signal temporel 1 (filtré)")
                axes[0, 0].grid(True)
            else:
                 axes[0, 0].set_title("Signal temporel 1 (filtré) - Données manquantes")
                 axes[0, 0].text(0.5, 0.5, "Données non disponibles", horizontalalignment='center', verticalalignment='center', transform=axes[0, 0].transAxes)


            # Spectre 1
            if len(freqs_pos1) > 0:
                axes[0, 1].stem(freqs_pos1, magnitude_pos1, basefmt=" ")
                axes[0, 1].set_xlabel("Fréquence (Hz)")
                axes[0, 1].set_ylabel("Amplitude")
                axes[0, 1].set_title("Spectre de Fourier (FFT) - Signal 1")
                axes[0, 1].set_xlim(0, 10)  # Zoom sur les basses fréquences
                axes[0, 1].grid(True)

                # Add annotations for prominent frequencies for Signal 1
                if prominent_freqs1:
                    for freq, mag in prominent_freqs1:
                        axes[0, 1].annotate(f'{freq:.2f} Hz', xy=(freq, mag), xytext=(freq + 0.1, mag + 0.01),
                                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=9)
            else:
                axes[0, 1].set_title("Spectre de Fourier (FFT) - Signal 1 - Données manquantes")
                axes[0, 1].text(0.5, 0.5, "Données non disponibles", horizontalalignment='center', verticalalignment='center', transform=axes[0, 1].transAxes)


            # Signal temporel 2
            if len(time_filtered2) > 1:
                 axes[1, 0].plot(time_filtered2, signal_filtered2, label="Signal 2 filtré", color='orange')
                 axes[1, 0].set_xlabel("Temps (s)")
                 axes[1, 0].set_ylabel("Amplitude")
                 axes[1, 0].set_title("Signal temporel 2 (filtré)")
                 axes[1, 0].grid(True)
            else:
                 axes[1, 0].set_title("Signal temporel 2 (filtré) - Données manquantes")
                 axes[1, 0].text(0.5, 0.5, "Données non disponibles", horizontalalignment='center', verticalalignment='center', transform=axes[1, 0].transAxes)


            # Spectre 2
            if len(freqs_pos2) > 0:
                axes[1, 1].stem(freqs_pos2, magnitude_pos2, basefmt=" ", linefmt='orange', markerfmt='o', label="Signal 2")
                axes[1, 1].set_xlabel("Fréquence (Hz)")
                axes[1, 1].set_ylabel("Amplitude")
                axes[1, 1].set_title("Spectre de Fourier (FFT) - Signal 2")
                axes[1, 1].set_xlim(0, 10)  # Zoom sur les basses fréquences
                axes[1, 1].grid(True)

                # Add annotations for prominent frequencies for Signal 2
                if prominent_freqs2:
                    for freq, mag in prominent_freqs2:
                        axes[1, 1].annotate(f'{freq:.2f} Hz', xy=(freq, mag), xytext=(freq + 0.1, mag + 0.01),
                                         arrowprops=dict(facecolor='black', shrink=0.05), fontsize=9, color='orange')
            else:
                axes[1, 1].set_title("Spectre de Fourier (FFT) - Signal 2 - Données manquantes")
                axes[1, 1].text(0.5, 0.5, "Données non disponibles", horizontalalignment='center', verticalalignment='center', transform=axes[1, 1].transAxes)


            plt.tight_layout()
            st.pyplot(fig)

            # Display fundamental frequencies
            st.write("### Fréquences Fondamentales")
            st.write(f"**Signal 1 :** {'{:.4f} Hz'.format(fundamental_frequency1) if fundamental_frequency1 != 0 else 'Non détectée'}")
            st.write(f"**Signal 2 :** {'{:.4f} Hz'.format(fundamental_frequency2) if fundamental_frequency2 != 0 else 'Non détectée'}")

            # Display noise power
            st.write("### Puissance de Bruit (1-10 Hz, hors fondamentale)")
            st.write(f"**Signal 1 :** {noise_power1:.4f}")
            st.write(f"**Signal 2 :** {noise_power2:.4f}")


            # Display prominent harmonics
            st.write("### Harmoniques Proéminentes - Signal 1")
            if len(prominent_freqs1) > 1:
                 other_prominent_freqs1 = [f for f in prominent_freqs1 if abs(f[0] - fundamental_frequency1) > 1e-9]
                 if other_prominent_freqs1:
                     for freq, mag in other_prominent_freqs1:
                         st.write(f"- Fréquence: {freq:.4f} Hz, Amplitude: {mag:.4f}")
                 else:
                      st.write("Aucune autre harmonique proéminente trouvée (au-delà du seuil d'affichage).")
            elif len(prominent_freqs1) == 1 and fundamental_frequency1 != 0:
                 st.write("Aucune autre harmonique proéminente trouvée (au-delà du seuil d'affichage).")
            else:
                 st.write("Aucune harmonique proéminente trouvée.")

            st.write("### Harmoniques Proéminentes - Signal 2")
            if len(prominent_freqs2) > 1:
                 other_prominent_freqs2 = [f for f in prominent_freqs2 if abs(f[0] - fundamental_frequency2) > 1e-9]
                 if other_prominent_freqs2:
                     for freq, mag in other_prominent_freqs2:
                         st.write(f"- Fréquence: {freq:.4f} Hz, Amplitude: {mag:.4f}")
                 else:
                      st.write("Aucune autre harmonique proéminente trouvée (au-delà du seuil d'affichage).")
            elif len(prominent_freqs2) == 1 and fundamental_frequency2 != 0:
                 st.write("Aucune autre harmonique proéminente trouvée (au-delà du seuil d'affichage).")
            else:
                 st.write("Aucune harmonique proéminente trouvée.")


            # Display comparison result
            st.write("### Conclusion de la comparaison")
            st.write(comparison_result)
             st.write(comparison_result2)

            # Add download button for prominent frequencies
            all_prominent_freqs = []
            if prominent_freqs1:
                all_prominent_freqs.extend(prominent_freqs1)
            if prominent_freqs2:
                all_prominent_freqs.extend(prominent_freqs2)


            if all_prominent_freqs:
                prominent_freqs_df = pd.DataFrame(all_prominent_freqs, columns=['Frequency (Hz)', 'Magnitude'])
                # Add a column to indicate which signal the frequency belongs to
                signal_indicators = []
                if prominent_freqs1:
                     signal_indicators.extend(['Signal 1'] * len(prominent_freqs1))
                if prominent_freqs2:
                     signal_indicators.extend(['Signal 2'] * len(prominent_freqs2))

                prominent_freqs_df['Signal'] = signal_indicators

                csv_data = prominent_freqs_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Télécharger les fréquences proéminentes (CSV)",
                    data=csv_data,
                    file_name='prominent_frequencies.csv',
                    mime='text/csv',
                )


        else:
            st.warning("Pas assez de points de données après application des seuils temporels pour effectuer l'analyse FFT.")


    except Exception as e:
        st.error(f"Erreur lors de la lecture des fichiers ou de l'analyse FFT : {e}")

else:
    st.info("Veuillez télécharger les deux fichiers CSV pour commencer l'analyse.")
