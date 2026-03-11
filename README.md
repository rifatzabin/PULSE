# PULSE

This repository contains the implementation of **PULSE**, a pipeline for Wi-Fi CSI-based sensing using temporal feature learning.


# Repository Structure

```text
PULSE/
│
├── datagen_temporal_csi.py      # Generate temporal CSI features
├── models_1dcnn.py              # CNN backbone and model heads
│
├── train_cnn.py                 # Train baseline CNN
├── eval_cnn.py                  # Evaluate trained CNN
│
├── pretrain_frel_cnn.py         # Representation pretraining (CE + SupCon)
├── adapt_frel_fewshot.py        # Few-shot adaptation on new environments
│
└── README.md
```

Clone the repository with ```git@github.com:rifatzabin/PULSE.git```



If you find the project useful and you use this code, please cite our paper:

```
@ARTICLE{11373292,
  author={Zabin, Rifat and Alam, Md. Golam Rabiul},
  journal={IEEE Wireless Communications Letters}, 
  title={PULSE: Physics-Aware Temporal Embedding Learning for Domain Adaptive Wireless Sensing}, 
  year={2026},
  volume={15},
  number={},
  pages={1752-1756},
  keywords={Sensors;Feature extraction;Wireless sensor networks;Wireless communication;Tensors;Training;Real-time systems;Frequency response;Computational modeling;Wireless fidelity;Wireless sensing;CFR;Domain adaptation},
  doi={10.1109/LWC.2026.3662002}}
  ```

  ## Download Dataset
  This project uses the **Beamsense Dataset**. The dataset can be found [here](https://github.com/kfoysalhaque/BeamSense).





# PIPELINE

### To generate temporal CSI features execute
```bash
python datagen_temporal_csi.py --raw ../CSI/Classroom --out ../Features/Classroom --window 64 --stride 64 --bands 8 --log-energy
```

This script converts CSI matrices into temporal feature windows used for training.

### To train the baseline CNN model execute
```bash
python train_cnn.py --features-dir ../Features/Classroom --ckpt-dir ../Checkpoints/Classroom/CNN
```

### To evaluate the trained CNN model execute
```bash
python eval_cnn.py --features-dir ../Features/Kitchen --ckpt-dir ../Checkpoints/Classroom/CNN
```

### Representation Learning
To pretrain the encoder using contrastive learning execute
```bash
python pretrain_frel_cnn.py --features-dir ../Features/Classroom --ckpt-dir ../Checkpoints/Classroom/FREL
```

# New Domain Adaptation
To adapt the pretrained encoder to a new environment execute
```bash
python adapt_frel_fewshot.py --target-features-dir ../Features/Kitchen --ckpt-dir ../Checkpoints/Classroom/FREL --shots-per-class 5
```



### For any question or query, please contact [Rifat Zabin](https://rifatzabin.github.io) or [rifat.zabin@g.bracu.ac.bd](mailto:rifat.zabin@g.bracu.ac.bd)


