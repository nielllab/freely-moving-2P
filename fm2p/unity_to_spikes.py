# Topdown and IMU data goes into Unity to get head-centered video.
# Head-centered view goes into Elliott's shifter network to get a
# shifted image. Then, take the shifted image and use a simple
# two-layer network to predict the neural activity of the population.
# The neural activity will be trained end-to-end with the shifter
# network.