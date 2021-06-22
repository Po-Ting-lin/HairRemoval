Welcome to collaborate and optimize this repo!

## HairRemoval
Implementation of hair removal in CUDA. The methods of hair detection refer to the paper[1][2].

### Requirements
* OpenCV 4.2.0
* CUDA 10.2

## Test Result  

Method            |Image Size     | Elapsed Time
------------------|:-------------:|--------------
Naive             |1.2 Megapixels |  18789 ms
GPU, OpenMp, AVX  |1.2 Megapixels |  325 ms

<!-- * image  
![](/sample/test2.png)

* mask  
![](/sample/test_mask2.png)

* result    
![](/sample/result2.png) -->

## References

```
[1] Pathan, S., Prabhu, K. & Siddalingaswamy, P.C.
    Hair detection and lesion segmentation in dermoscopic images using domain knowledge.
    Med Biol Eng Comput 56, 2051–2065 (2018). https://doi.org/10.1007/s11517-018-1837-9
[2] Adie, Heronimus Tresy Renata, and Ignatius Aldi Pradana. 
    "Parallel computing accelerated image inpainting using gpu cuda, theano, and tensorflow."
    2018 10th International Conference on Information Technology and Electrical Engineering (ICITEE). IEEE, 2018.
```
