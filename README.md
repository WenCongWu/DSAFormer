# Dual Sparse Aggregation Transformer for Multispectral Object Detection
This is an official PyTorch implementation for our DSAFormer. ArXiv paper will be download in [DSAFormer](https://arxiv.org/abs/2606.31015).

Our paper has been accepted by **IEEE Transactions on Circuits and Systems for Video Technology** for publication as a regular paper. The final version of the paper will be updated soon.

### 1. Dependences
 Create a conda virtual environment and activate it.
 1) conda create --name MOD python=3.9
 2) conda activate MOD
 3) pip install -r requirements.txt

### 2. Datasets download
Download these datasets and create a dataset folder to hold them.
1) FLIR dataset: [FLIR](https://drive.google.com/file/d/1o9lchkdQcPaYqqEa_d_6l3QewyfkDTCx/view?usp=drive_link)
2) LLVIP dataset: [Official Website](https://bupt-ai-cz.github.io/LLVIP/) or [Baidu Netdisk](https://pan.baidu.com/s/1BddW1CQV0z9PbWUEmUvGiQ?pwd=wuwc)
3) M3FD dataset: [M3FD](https://drive.google.com/file/d/1FSfAQQ80UvwE7mXKDAxZZnabUrsM9HHD/view?usp=drive_link)
4) MFAD dataset: [MFAD](https://drive.google.com/file/d/1BusF0_NY3pahjZLTqwXYYcKWQ8nSXy97/view?usp=drive_link)

### 3. Pretrained weights
Download our DSAFormer weights and create a weights folder to hold them.
1) FLIR dataset: [DSAFormer_FLIR.pt](https://drive.google.com/file/d/1sHWnmO-y-H-pzcs3Eb3wNDQykpAwFj6t/view?usp=drive_link)
2) LLVIP dataset: [DSAFormer_LLVIP.pt](https://drive.google.com/file/d/1XRYaiVPkmF7sCqw8b4t1_Xe7wihh6xyj/view?usp=drive_link)
3) M3FD dataset: [DSAFormer_M3FD.pt](https://drive.google.com/file/d/1rn5ktP5vkXp2mnOes8WUpkhDch8dUyWr/view?usp=drive_link)
4) MFAD dataset: [DSAFormer_MFAD.pt](https://drive.google.com/file/d/190-y7YWilP_-m-goCylUm7yZVay563Tt/view?usp=drive_link)

### 4. Training our DSAFormer
Dataset path, GPU, batch size, etc., need to be modified according to different situations.
```
python train.py
```

### 5. Test our DSAFormer

```
python test.py
```

### 6. Citation
If you find DSAFormer helpful for your research, please consider citing our work.
```BibTex
@article{Wu2026,
  title={Dual Sparse Aggregation Transformer for Multispectral Object Detection}, 
  author={Wencong Wu and Xiuwei Zhang and Hanlin Yin and Hongxi Zhang and Yanning Zhangg},
  journal={arXiv preprint arXiv:2606.31015},
  year={2026}
}
```

