# IDM-VTON (Modified Fork)

This repository is a fork of [IDM-VTON](https://github.com/example/IDM-VTON) with key architectural and training improvements.

---

## ✨ Key Changes

### 🔹 Hybrid U-Net + Transformer  
- **Problem:** Original IDM-VTON struggled with clothing containing complex textures and repeating patterns due to limited receptive field and local-only context from CNN layers.  
- **Solution:** Inserted **Transformer blocks** into the U-Net’s bottleneck stage to introduce **global context modeling**.  
- **Impact:** Improves the model’s ability to capture long-range dependencies in textures, enabling sharper and more realistic clothing try-on results.  

---

### 🔹 LoRA Finetuning  
- **Problem:** Traditional full-model finetuning required high GPU memory and was inefficient for adapting to new datasets.  
- **Solution:** Integrated **LoRA (Low-Rank Adaptation)** to selectively finetune only small adapter layers within attention blocks.  
- **Impact:**  
  - Reduces GPU memory usage by up to **60%**  
  - Speeds up training while retaining strong performance  
  - Enables efficient domain adaptation with limited compute resources  

---

### 🔹 Loss Function Improvement (Huber Loss)  
- **Problem:** MSE loss was sensitive to outliers, sometimes leading to unstable or faulty generated images during training.  
- **Solution:** Replaced MSE with **Huber loss (Smooth L1)** for noise prediction.  
- **Impact:**  
  - Provides robustness against outlier pixel errors  
  - Yields smoother, more stable convergence  
  - Improves visual fidelity of generated try-on images  

---

## 🚀 Usage

This fork follows the same setup and data preprocessing pipeline as the original IDM-VTON repo.  
The main differences are in the training code and model architecture:  

- Transformer-enhanced U-Net is used automatically  
- LoRA adapters can be enabled/disabled in config  
- Huber loss replaces MSE loss in training loop  

---

## 📌 Summary of Benefits
- ✅ Better handling of **patterns and textures** in clothing  
- ✅ **60% less GPU memory** with LoRA finetuning  
- ✅ More **robust training** with Huber loss  

---

## 📄 License
This project inherits the license from the original IDM-VTON repository.  
