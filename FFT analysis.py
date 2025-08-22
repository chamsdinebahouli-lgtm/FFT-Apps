import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

st.title("Application d'analyse FFT")

uploaded_file = st.file_uploader("Chargez votre fichier CSV", type=["csv"])

start_threshold = st.number_input("Exclure les N premières secondes :", min_value=0.0, value=30.0, step=1.0)
end_threshold = st.number_input("Exclure les N dernières secondes :", min_value=0.0, value=20.0, step=1.0)

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file into a pandas DataFrame
        # Assuming the CSV uses comma as decimal separator, similar to the original notebook
        df = pd.read_csv(uploaded_file, decimal=',')

        # Display a success message
        st.success("Fichier chargé avec succès !")

        # Optionally display the head of the DataFrame for confirmation
        st.subheader("Aperçu des données")
        st.dataframe(df.head())

        # Ensure 'Time' and 'Signal' columns exist
        if 'Time' in df.columns and 'Signal' in df.columns:
            time = df['Time'].values
            signal = df['Signal'].values

            # Apply the time slicing based on user input
            time_threshold_start = start_threshold
            time_threshold_end_abs = time[-1] - end_threshold # Calculate absolute end time threshold

            start_index = np.argmax(time >= time_threshold_start)
            # Find the index where time is less than or equal to the absolute end threshold
            # Search from the end of the array backwards
            # Ensure we don't go out of bounds if end_threshold is very large
            end_index_candidate = len(time) - 1 - np.argmax(time[::-1] <= time_threshold_end_abs)

            # Check if end_index_candidate is a valid index in the original time array
            if 0 <= end_index_candidate < len(time):
                 end_index = end_index_candidate
            else:
                 # If the calculated end_index is out of bounds, it means the end_threshold
                 # is larger than the available data duration after the start_threshold.
                 # In this case, we should end at the last data point available after start_index.
                 end_index = len(time) - 1 # Set end_index to the last index of the original array
                 if end_index < start_index: # Ensure start_index is not beyond the new end_index
                      start_index = end_index # If it is, set start_index to end_index for an empty range


            # Apply the slicing
            # Ensure end_index is after start_index
            if end_index >= start_index:
                time_filtered = time[start_index:end_index+1]
                signal_filtered = signal[start_index:end_index+1]
                st.write(f"Analyse des données de {time_filtered[0]:.2f} à {time_filtered[-1]:.2f} secondes.")
            else:
                st.warning("La plage temporelle spécifiée est invalide (le temps de fin n'est pas après le temps de début ou pas de données après le temps de début). Veuillez ajuster les seuils.")
                time_filtered = np.array([])
                signal_filtered = np.array([])


            if len(time_filtered) > 1:
                # === Parameters ===
                dt = time_filtered[1] - time_filtered[0]
                fs = 1 / dt

                # === FFT ===
                signal_centered = signal_filtered - np.mean(signal_filtered)
                fft_vals = np.fft.fft(signal_centered)
                freqs = np.fft.fftfreq(len(signal_centered), d=dt)

                # On garde les fréquences positives
                mask = freqs >= 0
                freqs_pos = freqs[mask]
                magnitude_pos = np.abs(fft_vals[mask]) / len(signal_centered)

                # === Find fundamental frequency and prominent harmonics ===
                fundamental_frequency = 0
                prominent_freqs = []
                if len(magnitude_pos) > 1:
                    # Find the index of the maximum magnitude, excluding the DC component (index 0)
                    fundamental_freq_index = np.argmax(magnitude_pos[1:]) + 1 # +1 to account for the excluded DC component
                    fundamental_frequency = freqs_pos[fundamental_freq_index]
                    prominent_freqs.append((fundamental_frequency, magnitude_pos[fundamental_freq_index]))


                # Find other prominent frequencies (harmonics)
                # Sort the positive frequencies by magnitude in descending order, excluding the DC component
                sorted_indices = np.argsort(magnitude_pos[1:])[::-1] + 1 # +1 to exclude DC component

                num_harmonics_to_display = 5 # You can change this number

                displayed_harmonics_count = 0
                for i in range(min(len(sorted_indices), len(freqs_pos) - 1)): # Ensure we don't go out of bounds and exclude DC
                     freq = freqs_pos[sorted_indices[i]]
                     mag = magnitude_pos[sorted_indices[i]]

                     # Avoid re-listing the fundamental and only display a limited number of other prominent freqs
                     if abs(freq - fundamental_frequency) > 1e-9 and displayed_harmonics_count < num_harmonics_to_display:
                         prominent_freqs.append((freq, mag))
                         displayed_harmonics_count += 1


                st.subheader("Résultats de l'analyse")

                # === Tracés ===
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Signal temporel
                axes[0].plot(time_filtered, signal_filtered, label="Signal filtré")
                axes[0].set_xlabel("Temps (s)")
                axes[0].set_ylabel("Amplitude")
                axes[0].set_title("Signal temporel (filtré)")
                axes[0].grid(True)

                # Spectre
                axes[1].stem(freqs_pos, magnitude_pos, basefmt=" ")
                axes[1].set_xlabel("Fréquence (Hz)")
                axes[1].set_ylabel("Amplitude")
                axes[1].set_title("Spectre de Fourier (FFT)")
                axes[1].set_xlim(0, 10)  # Zoom sur les basses fréquences
                axes[1].grid(True)

                # Add annotations for prominent frequencies
                for freq, mag in prominent_freqs:
                    axes[1].annotate(f'{freq:.2f} Hz', xy=(freq, mag), xytext=(freq + 0.1, mag + 0.01),
                                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=9)


                plt.tight_layout()
                st.pyplot(fig)

                # Display fundamental frequency
                if fundamental_frequency != 0:
                    st.write(f"**Fréquence fondamentale :** {fundamental_frequency:.4f} Hz")
                else:
                    st.write("**Fréquence fondamentale :** Non détectée (pas assez de données ou pas de pic significatif).")


                # Display prominent harmonics
                st.write("**Autres fréquences proéminentes (harmoniques potentielles) :**")
                if len(prominent_freqs) > 1: # Check if there are other frequencies besides the fundamental (if found)
                     # Filter out the fundamental frequency if it was added as the first element
                    other_prominent_freqs = [f for f in prominent_freqs if abs(f[0] - fundamental_frequency) > 1e-9]

                    if other_prominent_freqs:
                        for freq, mag in other_prominent_freqs:
                             st.write(f"- Fréquence: {freq:.4f} Hz, Amplitude: {mag:.4f}")
                    else:
                         st.write("Aucune autre harmonique proéminente trouvée (au-delà du seuil d'affichage).")
                elif len(prominent_freqs) == 1 and fundamental_frequency != 0: # Only fundamental found
                     st.write("Aucune autre harmonique proéminente trouvée (au-delà du seuil d'affichage).")
                else:
                    st.write("Aucune harmonique proéminente trouvée.")


            else:
                st.warning("Pas assez de points de données après application des seuils temporels pour effectuer l'analyse FFT.")

        else:
            st.error("Le fichier CSV téléchargé doit contenir les colonnes 'Time' et 'Signal'.")

    except Exception as e:
        # Catch potential errors during file reading and display an informative error message
        st.error(f"Erreur lors de la lecture du fichier ou de l'analyse FFT : {e}")
    except Exception as e:
        # Catch potential errors during file reading and display an informative error message
        st.error(f"Erreur lors de la lecture du fichier ou de l'analyse FFT : {e}")
