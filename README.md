# CrossFlow
This is a PyTorch-based reimplementation of CrossFlow, as proposed in 

>  Flowing from Words to Pixels: A Framework for Cross-Modality Evolution

>  [Qihao Liu](https://qihao067.github.io/) | [Xi Yin](https://xiyinmsu.github.io/) | [Alan Yuille](https://cogsci.jhu.edu/directory/alan-yuille/) | [Andrew Brown](https://www.robots.ox.ac.uk/~abrown/) | [Mannat Singh](https://ai.meta.com/people/1287460658859448/mannat-singh/)

> [[project page](https://cross-flow.github.io/)] | [[paper]()] | [[arxiv]()]

![teaser](https://github.com/qihao067/CrossFlow/blob/main/imgs/teaser.jpg)

This repository provides a **PyTorch-based reimplementation** of CrossFlow, with the following differences compared to the original paper:

- **Model Architecture**: The original paper utilizes DiMR as the model architecture. In contrast, this codebase supports training and inference with both [DiT](https://github.com/facebookresearch/DiT) (ICCV 2023, a widely adopted architecture) and [DiMR](https://github.com/qihao067/DiMR) (NeurIPS 2024, a state-of-the-art architecture).
- **Dataset**: The original model was trained on proprietary 350M dataset. In this implementation, the models are trained on open-source datasets, including [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/) and [JourneyDB](https://journeydb.github.io/).
- **LLMs**: The original model only supports CLIP as the language model, whereas this implementation includes models with CLIP and T5-XXL.

______

## TODO

- [x] Release inference code and 512px CLIP DiMR model.
- [ ] Release training code and a detailed training tutorial (ETA: Dec 20).
- [ ] Release all pretrained checkpoints, including:   (ETA: Dec 23)
  - 256px CLIP DiMR model, 
  - 256px T5XXL DiMR model, 
  - 512px T5XXL DiMR model, 
  - 512px T5XXL DiT model
- [ ] Provide a demo via Hugging Face Space and Colab.

______

## Terms of use

The project is created for research purposes.

______

## 
