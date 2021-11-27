# Rocknessssss_Gesture

This is a game controller that can play the game by facial reflex. The Components I used are Face detection, Facial landmark detection, 
Gesture Classification with thresholding on Tensorflow Deep CNN.

Here are the controls:

         Arrow keys (->, <-) = Mouth (smile, teeth)
         Button A = (Blink) Left eye 
         Button B = (Blink) Right eye
       
I created this model just for fun.

-> Problem Faced:-

    ● This model used Deep CNN to recognize face organs gesture, it failed to
      detect organs in blurry and little dark vision.

![Screenshot (259)](https://user-images.githubusercontent.com/75822824/119606639-8f6d6f00-be10-11eb-95e6-9768e964af6a.png) 



-> The Solution:-

    ● For better vision I used Binary and Truncate Thresholding after
      converting the BGR image to a Gray colored image, the machine can
      classify these images better than previous images. This improves the
      performance of the model
      
![Screenshot (258)](https://user-images.githubusercontent.com/75822824/119606706-a9a74d00-be10-11eb-8808-6df26b79f6df.png)


-> Introduction to Facial landmark detection (video)
https://drive.google.com/file/d/1lzY8jeQY6b5OQ70xVXrOKBWS37P5_B4r/view?usp=sharing

