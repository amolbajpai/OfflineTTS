import torch
import soundfile as sf

# Load the model
language = 'en'
model_id = 'v3_en'
speaker = 'en_0'
device = torch.device('cpu')

model, example_text = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language=language,
    speaker=model_id
)
model.to(device)

# Convert text to speech
text = """
Responsibilities:

    Write effective, scalable code.
    Develop back-end components to improve responsiveness and overall performance.
    Integrate user-facing elements into applications.
    Test and debug programs.
    Improve the functionality of existing systems.
    Assess and prioritize feature requests.
    Coordinate with customers to understand user requirements and provide technical solutions.
    Coordinate and support the team to ensure timely delivery."""
audio = model.apply_tts(
    text=text,
    speaker=speaker,
    sample_rate=48000
)

# Save using soundfile instead of torchaudio
sf.write('output.wav', audio.numpy(), 48000)
