# Changelog

## Train/Validation Split Update

### Changes Made

1. **Added Train/Validation Split**
   - New argument: `--val-split` (default: 0.2 = 20% validation)
   - Both demonstration and agent data are split into train and validation sets
   - Split is performed with a fixed random seed for reproducibility

2. **Separate Training and Validation Metrics**
   - Training is performed only on the training set
   - Validation set is evaluated during logging intervals (no gradients)
   - All metrics are now logged separately for train and validation

3. **Updated TensorBoard Logging**
   - **Training metrics** (prefix `train/`):
     - `train/loss`, `train/loss_demo`, `train/loss_agent`
     - `train/demo_acc`, `train/agent_acc`
     - `train/demo_logit_mean`, `train/agent_logit_mean`
     - `train/grad_penalty`
   
   - **Validation metrics** (prefix `val/`):
     - `val/loss`, `val/loss_demo`, `val/loss_agent`
     - `val/demo_acc`, `val/agent_acc`
     - `val/demo_logit_mean`, `val/agent_logit_mean`

4. **Enhanced Console Output**
   - Training loop now prints both train and validation metrics
   - Final evaluation reports metrics for both train and validation sets
   - Data split information shown during loading

5. **Updated Documentation**
   - README.md updated with new metrics descriptions
   - Added guidance on interpreting train/validation performance gaps
   - Added overfitting detection tips

### Benefits

- **Better evaluation**: Can now assess if discriminator generalizes to unseen data
- **Overfitting detection**: Large train/val gap indicates overfitting
- **Hyperparameter tuning**: Validation metrics guide hyperparameter selection
- **Statistical rigor**: More reliable performance estimates

### Example Output

```
TRAIN Loss: 0.1234 (Demo: 0.0567, Agent: 0.0667)
TRAIN Accuracy: Demo=0.987, Agent=0.982
VAL Loss: 0.1456 (Demo: 0.0678, Agent: 0.0778)
VAL Accuracy: Demo=0.975, Agent=0.968
```

### Usage

Default 80/20 split:
```bash
python verify_discriminator.py
```

Custom split (e.g., 70/30):
```bash
python verify_discriminator.py --val-split 0.3
```

No validation (use all data for training):
```bash
python verify_discriminator.py --val-split 0.0
```
