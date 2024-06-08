import soundfile as sf
import numpy as np
import librosa as lb
import torchcrepe
import torch
from ddx7.core import _DB_RANGE,_REF_DB

_CREPE_WIN_LEN = 1024
HOP_SIZE = 64

def calc_f0_ref(audio, rate, hop_size,fmin,fmax,model,
            batch_size,device,center=False):
    if center is False:
      # Add padding to the end. Then execute crepe w/o padding.
      # Crepe pads so that the signal stays in the center.
      n_samples_initial = int(audio.shape[-1])
      n_frames = int(np.ceil(n_samples_initial / hop_size))
      n_samples_final = (n_frames - 1) * hop_size + _CREPE_WIN_LEN
      pad = n_samples_final - n_samples_initial
      audio = np.pad(audio, ((0, pad),), "constant")


    audio = torch.from_numpy(audio).unsqueeze(0).float().to(device)
    crepe_tuple = torchcrepe.predict(audio,
                        rate,
                        hop_size,
                        fmin,
                        fmax,
                        model,
                        return_periodicity=True,
                        batch_size=batch_size,
                        device=device,
                        pad=center)
    
    f0 = crepe_tuple[0]
    confidence = crepe_tuple[1]
    if center is True:
      f0 = f0[:,0:-1] #Discard the last sample
      confidence = confidence[:,0:-1] #Discard the last sample

    f0 = f0.squeeze(0).cpu().numpy()
    confidence = confidence.squeeze(0).cpu().numpy()

    return f0,confidence
    
def calc_f0_pyin(audio, fmin, fmax, sr, frame_length, win_length, hop_length,
                 center=False):
    f0 = lb.pyin(audio, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, win_length=win_length, hop_length=hop_length, center=center, fill_na=None)
    return f0
    
def test_f0(audio, fmin, fmax, sr, frame_length, win_length, hop_length, batch_size, device, center=False):
    # Esta es la referencia, para que sepan la forma que tiene que tener el dato de salida y los valores de f0 que debería darles (aproximadamente)
    f0_crepe_ref = calc_f0_ref(audio, sr, hop_length, fmin, fmax, 'full', batch_size, device, center)[0]
    # Acá viene su propuesta, usen esto para verificar con un audio de prueba que su algoritmo respeta la forma del dato de salida y que los valores que les da no estén muy lejos de la referencia
  #  f0_propuesta = calc_f0_propuesta(args_in)
    # Acá viene su propuesta para el cálculo de loudness. Recuerden revisar los comentarios que dejé en la función correspondiente.
  #  loud_propuesta = calc_loudness(args_in)
    
#def calc_loudness(audio, rate, n_fft=_LD_N_FFT, hop_size=64,
 #                 range_db=_DB_RANGE,ref_db=_REF_DB,center=False):
    # acá viene su función de cálculo de loudness, datos importantes:
    # *el resultado que sale de acá tiene largo igual a la cantidad de frames que representan al audio de entrada.
    # *el resultado está en dBs normalizados y ponderados A (arranquen por ahí, después prueben no ponderar o ponderar distinto a ver qué pasa). 

#f0 = test_f0(audio, 50, 2000, sr, 2048, 1024, HOP_SIZE, 128, 'cpu')
 
audio, sr = sf.read("test_audio.wav")
print(len(audio))
audio = audio[1500000:1600000]
print(len(audio))

#f0_crepe = calc_f0_ref(audio, sr, HOP_SIZE, 50, 2000, "full", 128, 'cpu')
#f0_pyin = calc_f0_pyin(audio, 50, 2000, sr, 2048, 1024, HOP_SIZE)

_RMS_FRAME = 2048
_CREPE_WIN_LEN = 1024
_LD_N_FFT = 2048

def calc_loudness(audio, rate, n_fft=_LD_N_FFT, hop_size=64,
                  range_db=_DB_RANGE,ref_db=_REF_DB,center=False):
    np.seterr(divide='ignore')

    """Compute loudness, add to example (ref is white noise, amplitude=1)."""
    # Copied from magenta/ddsp/spectral_ops.py
    # Get magnitudes.
    if center is False:
        # Add padding to the end
        n_samples_initial = int(audio.shape[-1])
        n_frames = int(np.ceil(n_samples_initial / hop_size))
        n_samples_final = (n_frames - 1) * hop_size + n_fft
        pad = n_samples_final - n_samples_initial
        audio = np.pad(audio, ((0, pad),), "constant")
    spectra = librosa.stft(
        audio, n_fft=n_fft, hop_length=hop_size, center=center).T

    # Compute power
    amplitude = np.abs(spectra)
    amin = 1e-20  # Avoid log(0) instabilities.
    power_db = np.log10(np.maximum(amin, amplitude))
    power_db *= 20.0

    # Perceptual weighting.
    frequencies = librosa.fft_frequencies(sr=rate, n_fft=n_fft)
    a_weighting = librosa.A_weighting(frequencies)[np.newaxis, :]
    loudness = power_db + a_weighting

    # Set dynamic range.
    loudness -= ref_db
    loudness = np.maximum(loudness, -range_db)

    # Average over frequency bins. (loudness is taken from the fft dimension!)
    mean_loudness_db = np.mean(loudness, axis=-1)
    return mean_loudness_db.astype(np.float32)




