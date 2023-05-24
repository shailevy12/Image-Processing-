This code implements a latent optimization technique for image generation using a pretrained generative model (G).
It takes a target image as input and iteratively optimizes the latent vector (w_opt) to generate an image that matches the target.
The optimization process involves adjusting the latent vector and applying various degradation modes to the generated image.

The degradation modes supported by the code include inpainting (using a provided mask), grayscale degradation, and Gaussian blur.
By applying these degradation modes, the code allows for exploring the effects of different image degradations on the optimization process.

Overall, this code provides a flexible framework for latent optimization, enabling the generation of images that align
with a target while considering different degradation scenarios.
