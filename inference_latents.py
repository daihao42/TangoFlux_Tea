import torchaudio
from tangoflux import TangoFluxInference

model = TangoFluxInference(name="declare-lab/TangoFlux")
latents = model.generate_latents_batch(["Hammer slowly hitting the wooden table", "not a sun"], steps=50, duration=10)

print("Latents generated successfully:", latents[1].shape)

