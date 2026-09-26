"""
# Generating images, audio and video

Language models generate text one token at a time. Pictures, sound and video
are made differently: an autoencoder squeezes them into a small code and back,
a GAN pits a forger against a detective, and a diffusion model learns to turn
pure noise into an image by removing a little noise at a time. Multimodal
models then connect all of these to a language model, so one system can read
an image, hear speech and answer in words.
"""

from primer.curriculum import reading_list as _reading_list

__doc__ += _reading_list("primer.ml.generative")
