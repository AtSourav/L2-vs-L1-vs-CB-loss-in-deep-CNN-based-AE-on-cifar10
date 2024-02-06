# AE-n-VAE-with-CB-loss-on-stl10
Comparing performance of an AE trained using mse (L1) vs mse (L2) vs Continuous Bernoulli loss (based on 1907.06845) on the cifar10 dataset. 

The architecture chosen consists deep CNNs (and upsampling layers) inspired from the VGG networks and is as follows. The latent dim used is 200.

## Encoder:  (N,p) refers to a conv layer with N channels and a filter with edge p, valid padding.

Input -> (64,3) -> (64,2) -> (MaxPool, 2) -> (128,2) -> (128,2) -> (128,1) -> (128,2) -> (128,2) -> (128,1) -> (256,2) -> (256,2) -> (256,1) -> (256,2) -> (256,2) -> (256,1) -> (512,2) -> (512,2) -> (512,2) -> (512,2) -> (512,2) -> (512,1) -> Flatten -> Dense(3* latent_dim) -> Dense(2* latent_dim) -> Dense(latent_dim) -> Encoder output.

## Decoder: Includes deconv,conv and upsampling layers. (N,p)* refers to a deconv layer, valid padding.

Encoder output -> Dense(2* latent_dim) -> Dense(3* latent_dim) -> Dense(4* latent_dim) -> Dense(2* 2* 1024) -> Reshape(2,2,1024) -> (1024,1)* -> (512,1)* -> (512,2)* -> (512,2)* -> (256,1)* -> (256,2)* -> (256,2)* -> (256,2)* -> Upsampling(2,2) -> (128,2)* -> (128,2)* -> (128,2)* -> (128,2)* -> Upsampling(2,2) -> (128,2) -> (128,2) -> (128,1) -> (64,2) -> (64,2) -> (3,1) -> Decoder output.

## Results:

Between the reconstructions from the network trained w.r.t. L2 loss and that trained w.r.t. CB loss, the latter set has brighter colours (too bright as they are brighter than the originals) and very sharp contrast. This can be explained by the convex nature of the log norm of p(x) which pushes pixel values to extremes.

The results using L1 loss appear to be the worst as the reconstructed images are even more blurry than the ones corresponding to L2 loss although the colours are okay. 
