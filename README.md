# Vision Glasses Software App for Visually Impaired Individuals

This software is designed for a pair of smart glasses, using machine learning and a CRNN model to convert text in the user's field of view into audio in real time. It provides a hands-free, accessible reading experience for visually impaired students. While I’ve completed the core software, capable of capturing images, processing text, and generating speech, the hardware integration is part of my future plans. The final device will include a Raspberry Pi, camera module, and other compact components, forming lightweight, wearable glasses. Unlike traditional solutions like braille books or e-pens, this system offers a portable and easy to use solution for students in low-resource environments to access educational content.

To run the final glasses code, it needs to be executed on a Raspberry Pi Zero, but the model itself can be run independently, following the steps below:
  - download this repo
  - enter the training folder
  - run the inference.py file
  - this will give the test image to the model, and return the extracted text
  - this is still a work in progress, as I still need to add the audio/speech system to the inference file, and create this as a web app
