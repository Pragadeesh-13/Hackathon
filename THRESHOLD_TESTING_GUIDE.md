# Threshold-Based Breed Recognition

## Overview

The system now supports **confidence threshold-based recognition** that can distinguish between:
- ✅ **Positive identification** - Image confidently matches a known breed
- ❌ **Rejection/Unknown** - Image does not confidently match any known breed

This allows you to test with different images (Murrah buffalo, other animals, random objects) and get appropriate responses.

## How It Works

1. **Feature Extraction**: Extract deep features from the input image
2. **Similarity Computation**: Compute cosine similarity with breed prototypes  
3. **Threshold Check**: Compare best similarity score against confidence threshold
4. **Decision**: 
   - If score ≥ threshold → **Positive identification** with breed name
   - If score < threshold → **"Unknown/Not recognized"**

## Testing Your System

### 1. Command Line Testing

**Test with Murrah image (should be recognized):**
```bash
python infer.py dataset/Murrah/murrah_0002.jpg --confidence-threshold 0.6 --show-breed-info
```
Expected output: `[✓] POSITIVE IDENTIFICATION: Murrah`

**Test with high threshold (should be rejected):**
```bash
python infer.py dataset/Murrah/murrah_0002.jpg --confidence-threshold 0.98
```
Expected output: `[X] NO CONFIDENT MATCH`

**Test with non-Murrah image:**
```bash
python infer.py path/to/dog.jpg --confidence-threshold 0.6
```
Expected output: `[X] NO CONFIDENT MATCH` (assuming the dog image gets low similarity)

### 2. Web Interface Testing

1. **Start the app:**
   ```bash
   streamlit run app.py
   ```

2. **Configure threshold:**
   - Use the "Confidence Threshold" slider in the sidebar
   - Default: 0.6 (recommended for testing)
   - Higher values = stricter (fewer false positives)
   - Lower values = more lenient (more detections)

3. **Test different images:**
   - Upload Murrah buffalo images → Should show "✅ Positive Identification"
   - Upload other animal images → Should show "❌ Unknown / Not Recognized"
   - Upload random objects → Should show "❌ Unknown / Not Recognized"

## Recommended Threshold Values

| Threshold | Use Case | Description |
|-----------|----------|-------------|
| **0.8-0.9** | Production (Strict) | Very few false positives, may miss some valid images |
| **0.6-0.7** | **Recommended for Testing** | Good balance of accuracy and coverage |
| **0.4-0.5** | Research/Lenient | More detections, may have some false positives |
| **0.2-0.3** | Debug/Analysis | Very lenient, useful for understanding model behavior |

## Example Test Scenarios

### Scenario 1: Perfect Murrah Image
```
Image: High-quality Murrah buffalo photo
Threshold: 0.6
Expected: Confidence ~0.85-0.95 → ✅ POSITIVE IDENTIFICATION: Murrah
```

### Scenario 2: Poor Quality Image  
```
Image: Blurry or partial Murrah image
Threshold: 0.6
Expected: Confidence ~0.4-0.6 → ❌ NO CONFIDENT MATCH
```

### Scenario 3: Different Animal
```
Image: Dog, cat, horse, etc.
Threshold: 0.6  
Expected: Confidence ~0.1-0.4 → ❌ NO CONFIDENT MATCH
```

### Scenario 4: Non-Animal Object
```
Image: Car, building, food, etc.
Threshold: 0.6
Expected: Confidence ~0.0-0.3 → ❌ NO CONFIDENT MATCH
```

## Understanding the Output

### Positive Identification
```
[✓] POSITIVE IDENTIFICATION: Murrah
   Confidence: 0.953 (>= 0.600 threshold)

BREED INFORMATION: Murrah
Type: Buffalo
Region: Punjab, Haryana, Rajasthan
Average Milk Yield: 12-18 liters/day
Features: Large, black body with curled horns...
```

### Rejection/Unknown
```
[X] NO CONFIDENT MATCH
   Best match: Murrah (0.451)
   This is below the confidence threshold of 0.600
   The image likely does NOT show a Murrah or any recognized breed.
```

## Adding More Breeds

To enable multi-breed testing:

1. **Add breed folders:**
   ```
   dataset/
   ├── Murrah/          # Existing
   ├── Gir/             # Add cattle images here
   ├── Sahiwal/         # Add cattle images here  
   └── Holstein/        # Add cattle images here
   ```

2. **Rebuild prototypes:**
   ```bash
   python prototype.py --augment
   ```

3. **Test multi-breed recognition:**
   ```bash
   python infer.py gir_image.jpg --confidence-threshold 0.6
   # Should identify as Gir, not Murrah
   ```

## Tips for Testing

1. **Start with threshold 0.6** - Good balance for most testing
2. **Test with obvious non-Murrah images first** - Verify rejection works
3. **Try different Murrah images** - Check consistency
4. **Adjust threshold based on your needs:**
   - Too many false positives? → Increase threshold
   - Missing valid detections? → Decrease threshold
5. **Use the web interface** - More intuitive for interactive testing

## Technical Notes

- **Confidence scores range from 0.0 to 1.0**
- **Higher scores = more similar to breed prototype**
- **Scores are based on cosine similarity of deep features**
- **Even "wrong" images will get some similarity score** (hence the need for thresholds)
- **The system learns from training images, so image quality and lighting matter**

This threshold-based approach makes your system much more practical for real-world deployment where you need to handle diverse input images!