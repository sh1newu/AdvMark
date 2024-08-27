# AdvMark
The supplementary material for the paper: "Are Watermarks Bugs for Deepfake Detectors? Rethinking Proactive Forensics".  
Paper Link: https://doi.org/10.24963/ijcai.2024/673  

***Watermak signals are imperceptible to human eyes but can interfere with Deepfake detectors.***

**Reference only**  
Due to the numerous experiments conducted, we regrettably did not save all the code from the intermediate processes. Therefore, the code [here](https://github.com/sh1newu/AdvMark/tree/main/network) is for reference only. As you can see, the implementation is quite simple and consists of just a few lines. We expect that you can implement the method based on any existing proactive watermarking model and passive Deepfake detectors you prefer. In addition, we provide the [dataset](https://drive.google.com/drive/folders/1NKnkhh5102pPs8DP-2MZ74r7uzUVWZov?usp=sharing) and [detectors](https://drive.google.com/drive/folders/1771ni4ERqjGkwcj_FlHWJZK7wf9o7fEf?usp=sharing) used in the experiments.

**Acknowledgment**  
Thanks for the following helpful repositories:  
Image Watermarking: [MBRS](https://github.com/jzyustc/MBRS), [FaceSigns](https://github.com/paarthneekhara/FaceSignsDemo), [SepMark](https://github.com/sh1newu/SepMark), [FIN](https://github.com/QQiuyp/FIN), [PIMoG](https://github.com/FangHanNUS/PIMoG-An-Effective-Screen-shooting-Noise-Layer-Simulation-for-Deep-Learning-Based-Watermarking-Netw)   
Deepfake Detection: [Xception](https://github.com/ondyari/FaceForensics), [EfficientNet](https://github.com/ldz666666/Style-atk), [CNND](https://github.com/peterwang512/CNNDetection), [FFD](https://github.com/JStehouwer/FFD_CVPR2020), [PatchForensics](https://github.com/chail/patch-forensics), [MultiAtt](https://github.com/yoctta/multiple-attention), [RFM](https://github.com/crywang/RFM), [RECCE](https://github.com/VISION-SJTU/RECCE), [SBI](https://github.com/mapooon/SelfBlendedImages)   
Deepfake Generation: [SimSwap](https://github.com/neuralchen/SimSwap), [FOMM](https://github.com/AliaksandrSiarohin/first-order-model), [StarGAN](https://github.com/yunjey/stargan), [StyleGAN](https://github.com/NVlabs/stylegan)  
 
The code is strictly for non-commercial academic use only.  
Contact: xinliao@hnu.edu.cn / shinewu@hnu.edu.cn

If you find our work useful, please consider citing:  

```
@inproceedings{wu2024watermarks,  
  title={Are Watermarks Bugs for Deepfake Detectors? Rethinking Proactive Forensics},  
  author={Wu, Xiaoshuai and Liao, Xin and Ou, Bo and Liu, Yuling and Qin, Zheng},  
  booktitle={Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence},  
  year={2024}  
}
```
