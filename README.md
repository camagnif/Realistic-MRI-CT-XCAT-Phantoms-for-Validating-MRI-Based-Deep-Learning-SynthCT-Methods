# Realistic-MRI-CT-XCAT-Phantoms-for-Validating-MRI-Based-Deep-Learning-SynthCT-Methods

This repository contains code corresponding to the implementation described in:
Camagni, F., Nakas, A., & Paganelli, C. (2025). [Generation of multimodal realistic computational phantoms as a test-bed for validating deep learning-based cross-modality synthesis techniques.](https://doi.org/10.1007/s11517-025-03437-4)  
Please refer to the paper for detailed methodology and experimental context.

This repository contains code for generating realistic multimodal (i.e., CT and MRI) computational phantoms to validate AI models. Traditional phantoms often fail to capture the continuous, noisy, and heterogeneous characteristics of clinical data, causing artifacts and inflated error metrics when evaluating GAN-based models trained on patient data.

Key contributions:
- Creation of multimodal phantom datasets using CycleGANs that statistically resemble clinical data while retaining known ground truth.
- Evaluation of generated data through paired and unpaired assessments.
- Providing a controlled, realistic test set for synthetic CT generation models trained on real abdominal MRI, bridging the gap between synthetic and real domains.



