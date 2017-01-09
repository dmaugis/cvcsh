attempt to port matlab code in C++11 w/OpenCV3
http://www.eng.tau.ac.il/~simonk/CSH/

Coherency Sensitive Hashing (CSH)

Simon Korman and Shai Avidan

abstract

Coherency Sensitive Hashing (CSH) extends Locality Sensitivity Hashing (LSH) and PatchMatch to quickly find matching patches between two images. LSH relies on hashing, which maps similar patches to the same bin, in order to find matching patches. PatchMatch, on the other hand, relies on the observation that images are coherent, to propagate good matches to their neighbors, in the image plane. It uses random patch assignment to seed the initial matching. CSH relies on hashing to seed the initial patch matching and on image coherence to propagate good matches. In addition, hashing lets it propagate information between patches with similar appearance (i.e., map to the same bin). This way, information is propagated much faster because it can use similarity in appearance space or neighborhood in the image plane. As a result, CSH is at least three to four times faster than PatchMatch and more accurate, especially in textured regions, where reconstruction artifacts are most noticeable to the human eye. We verified CSH on a new, large scale, data set of 133 image pairs.


