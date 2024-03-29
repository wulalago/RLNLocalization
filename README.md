# RLNLocalization
This repository is the implementation of the cascaded network for Recurrent Laryngeal Nerve Localization by Haoran Dou in CISTIB at University of Leeds.

**Localizing the Recurrent Laryngeal Nerve via Ultrasound with a Bayesian Shape Framework.**  
*Haoran Dou, Luyi Han, Yushuang He, Jun Xu, Nishant Ravikumar, Ritse Mann, Alejandro F. Frangi, Pew-Thian Yap, Yunzhi Huang.*  
International Conference on Medical Image Computing and Computer Assisted Intervention, 2022. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_25) [[arXiv]](https://arxiv.org/pdf/2206.15254.pdf)

![framework](framework.png)
> Tumor infiltration of the recurrent laryngeal nerve (RLN) is a contraindication for robotic thyroidectomy and can be difficult to detect via standard laryngoscopy. Ultrasound (US) is a viable alternative for RLN detection due to its safety and ability to provide real-time feedback. However, the tininess of the RLN, with a diameter typically less than 3,mm, poses significant challenges to the accurate localization of the RLN. In this work, we propose a knowledge-driven framework for RLN localization, mimicking the standard approach surgeons take to identify the RLN according to its surrounding organs. We construct a prior anatomical model based on the inherent relative spatial relationships between organs. Through Bayesian shape alignment (BSA), we obtain the candidate coordinates of the center of a region of interest (ROI) that encloses the RLN. The ROI allows a decreased field of view for determining the refined centroid of the RLN using a dual-path identification network, based on multi-scale semantic information. Experimental results indicate that the proposed method achieves superior hit rates and substantially smaller distance errors compared with state-of-the-art methods. 

## Usage

### How to Run
Our method has five steps for training and testing.
1. Run `python train.py` to train the UNet for the segmentation of the CCA, thyroid, trachea.
2. Run `python infer.py` to obtain the coarse localization results of the RLN.
3. Run `python statistic_test.py` and `python prior_localize.py` to obtain the results of the Bayesian alignment.
4. Run `python refine_train.py` to train the multi-scale locator for the localization of RLN.
5. Run `python refine infer.py` to obtain the final localization results of RLN.

## Results
![Results](Result_h.png)

## Citation
If this work is helpful for you, please cite our paper as follows:
```
@article{dou2022localizing,
  title={Localizing the Recurrent Laryngeal Nerve via Ultrasound with a Bayesian Shape Framework},
  author={Dou, Haoran and Han, Luyi and He, Yushuang and Xu, Jun and Ravikumar, Nishant and Mann, Ritse and Frangi, Alejandro F and Yap, Pew-Thian and Huang, Yunzhi},
  journal={arXiv preprint arXiv:2206.15254},
  year={2022}
}
```
