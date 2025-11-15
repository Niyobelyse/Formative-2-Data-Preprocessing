# Voice Verification System Guide

This guide explains how to use the voice verification system for extracting features and training/testing models.

## Overview

The system consists of three main components:
1. **Feature Extraction** (`audio_feature_extraction.py`) - Extracts audio features from all persons
2. **Model Training** (`voice_verification_model.py`) - Trains multiple models and selects the best one
3. **Model Testing** (`test_voice_model.py`) - Tests the trained model with new audio samples

## Step 1: Extract Audio Features

Extract features from all audio files for each person. Features are saved separately for each person and then merged.

### Basic Usage

```bash
python audio_feature_extraction.py
```

### With Silence Removal

Removing silence can improve feature extraction quality:

```bash
python audio_feature_extraction.py --remove-silence
```

### Options

- `--remove-silence`: Remove silence from audio before feature extraction (recommended for better accuracy)
- `--base-path`: Base path to search for audio folders (default: current directory)

### Output

The script will:
1. Process audio files from:
   - `audio/augmented/` (Fidel's files)
   - `augmented_audio/Kerie/` (Kerie's files)
   - `augmented_audio/Irais/` (Irais's files)

2. Save separate feature files:
   - `features_audio/Fidel_audio_features.csv`
   - `features_audio/Kerie_audio_features.csv`
   - `features_audio/Irais_audio_features.csv`

3. Create merged features file:
   - `data/processed/audio_features_all.csv`

### Features Extracted

- **MFCCs** (78 features): Most important for voice recognition
  - 13 MFCC coefficients (mean + std)
  - 13 MFCC Delta (mean + std)
  - 13 MFCC Delta-Delta (mean + std)
- **Spectral Features** (6 features): Centroid, Roll-off, Bandwidth
- **Energy Features** (2 features): RMS Energy
- **Zero Crossing Rate** (2 features)
- **Chroma Features** (24 features): 12 pitch classes
- **Spectral Contrast** (14 features): 7 frequency bands
- **Tempo** (1 feature)

**Total: ~127 features per audio sample**

## Step 2: Train Voice Verification Model

Train multiple models (Random Forest, Logistic Regression, XGBoost) and automatically select the best one.

### Basic Usage

```bash
python voice_verification_model.py
```

### Options

- `--features`: Path to merged features CSV (default: `data/processed/audio_features_all.csv`)
- `--output-dir`: Directory to save trained models (default: `models/`)

### Output

The script will:
1. Load and prepare the features
2. Train three models:
   - **Random Forest**: Good for non-linear relationships, handles feature interactions well
   - **Logistic Regression**: Fast, interpretable, good baseline
   - **XGBoost**: Often best performance, handles complex patterns

3. Compare models and select the best one based on test accuracy

4. Save the best model:
   - `models/voice_verification_[model_name].pkl`
   - `models/scaler.pkl` (for feature scaling)
   - `models/feature_columns.txt` (feature column names)

### Model Selection

The system automatically selects the best model based on:
- **Test Accuracy**: Primary metric
- **Cross-Validation Score**: Ensures generalization

For voice verification, **XGBoost** typically performs best, but **Random Forest** is also excellent and more interpretable.

## Step 3: Test the Model

Test the trained model with new audio samples.

### Test Single File

```bash
python test_voice_model.py --audio path/to/audio.wav
```

### Test Folder

```bash
python test_voice_model.py --audio path/to/folder/
```

### With Silence Removal

```bash
python test_voice_model.py --audio path/to/audio.wav --remove-silence
```

### Options

- `--audio`: Path to audio file or folder to test (required)
- `--model`: Path to trained model file (default: `models/voice_verification_randomforest.pkl`)
- `--scaler`: Path to scaler file (default: `models/scaler.pkl`)
- `--features`: Path to feature columns file (default: `models/feature_columns.txt`)
- `--remove-silence`: Remove silence from audio before testing

### Output

The script will show:
- Predicted person
- Confidence scores for all persons
- Accuracy (if testing a folder with known labels)

## Complete Workflow Example

```bash
# 1. Extract features (with silence removal for better accuracy)
python audio_feature_extraction.py --remove-silence

# 2. Train models
python voice_verification_model.py

# 3. Test the model
python test_voice_model.py --audio audio/test_sample.wav --remove-silence
```

## Which Model Works Best for Voice Verification?

### Recommendation: **XGBoost** or **Random Forest**

**XGBoost** is typically the best choice because:
- Handles complex voice patterns well
- Good at capturing subtle differences between speakers
- Robust to overfitting
- High accuracy on voice verification tasks

**Random Forest** is also excellent because:
- More interpretable (can see feature importance)
- Faster training
- Good performance on voice data
- Less prone to overfitting with small datasets

**Logistic Regression** is useful as a baseline:
- Fast and simple
- Good for understanding linear relationships
- Less accurate but more interpretable

The system will automatically select the best model based on test accuracy.

## Silence Removal

### When to Use

**Recommended**: Use `--remove-silence` when:
- Audio files have long silence at the beginning/end
- Background noise is minimal
- You want cleaner feature extraction

**Not Recommended**: When:
- Silence contains important information
- Audio is already clean
- Very short audio samples (might remove too much)

### How It Works

Uses librosa's voice activity detection to trim silence from audio before feature extraction. This can improve accuracy by focusing on the actual voice content.

## Troubleshooting

### No features extracted

- Check that audio files are in the correct folders:
  - `audio/augmented/` for Fidel
  - `augmented_audio/Kerie/` for Kerie
  - `augmented_audio/Irais/` for Irais
- Ensure files are in `.wav` format

### Model training fails

- Check that `data/processed/audio_features_all.csv` exists
- Ensure you have at least 2 persons with multiple samples each
- Check that features were extracted correctly

### Low accuracy

- Try using `--remove-silence` for feature extraction
- Ensure you have enough training samples per person (recommended: 10+)
- Check audio quality (clear recordings work best)
- Try different models (the system will select the best automatically)

## File Structure

```
.
├── audio_feature_extraction.py    # Feature extraction module
├── voice_verification_model.py    # Model training script
├── test_voice_model.py            # Model testing script
├── features_audio/                 # Individual person features
│   ├── Fidel_audio_features.csv
│   ├── Kerie_audio_features.csv
│   └── Irais_audio_features.csv
├── data/processed/                 # Merged features
│   └── audio_features_all.csv
└── models/                         # Trained models
    ├── voice_verification_*.pkl
    ├── scaler.pkl
    └── feature_columns.txt
```

## Requirements

Make sure you have installed:
- `librosa` - Audio processing
- `scikit-learn` - Machine learning models
- `xgboost` - XGBoost classifier
- `pandas` - Data handling
- `numpy` - Numerical operations
- `joblib` - Model saving/loading

Install with:
```bash
pip install librosa scikit-learn xgboost pandas numpy joblib
```

