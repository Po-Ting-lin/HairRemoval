## HairRemoval
* hair detection

Implementation of hair removal in CUDA. The methods of hair detection refer to the paper[1].

Three versions of hair detection with their performance, respectively.  
![](/sample/performance.png)

* hair inpainting

### Requirements
* C++ STL
* OpenCV 4.2.0
* CUDA 10.2
* Enable OpenMp

## Demo  
* image  
![](/sample/raw.jpg)

* mask  
![](/sample/mask.jpg)

* result    
![](/sample/processed.jpg)

## References

```
[1] Pathan, S., Prabhu, K. & Siddalingaswamy,
    P.C. Hair detection and lesion segmentation in dermoscopic images using domain knowledge.
    Med Biol Eng Comput 56, 2051–2065 (2018). https://doi.org/10.1007/s11517-018-1837-9
```
