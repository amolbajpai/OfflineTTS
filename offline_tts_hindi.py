import torch
import soundfile as sf
from aksharamukha import transliterate

# Load the Indic model
language = 'indic'
model_id = 'v4_indic'
device = torch.device('cpu')

model, example_text = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language=language,
    speaker=model_id
)
model.to(device)

# Hindi text (in Devanagari script)
hindi_text = """जिम्मेदारियाँ:


प्रभावी, स्केलेबल कोड लिखें।
प्रतिक्रिया और समग्र प्रदर्शन में सुधार के लिए बैक-एंड घटक विकसित करें।
उपयोगकर्ता-सामना करने वाले तत्वों को अनुप्रयोगों में एकीकृत करें।
कार्यक्रमों का परीक्षण और डीबग करें।
मौजूदा सिस्टम की कार्यक्षमता में सुधार करें।"""

# Romanize the text from Devanagari to ISO format
roman_text = transliterate.process('Devanagari', 'ISO', hindi_text)
print("Romanized text:", roman_text)

# Convert text to speech using Hindi male or female speaker
audio = model.apply_tts(
    text=roman_text,
    speaker='hindi_male',  # or 'hindi_female'
    sample_rate=48000
)

# Save audio file
sf.write('hindi_output.wav', audio.numpy(), 48000)
