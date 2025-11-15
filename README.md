Formative 2

# Load model (ONE file!)

model = joblib.load('voice_models/voice_recognition_complete.pkl')

# Test audio features (NO filename!)

features = {
'mfcc_0_mean': -431.14,
'mfcc_1_mean': 85.68, # ... 43 audio features
}

# Predict

result = authenticate_voice(features)

# Result:

{
'person': 'Fidel',
'phrase': 'approve',
'confidence': 0.89,
'authenticated': True
}
