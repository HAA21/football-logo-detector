# football-logo-detector

I have implemented football club logo detector using TensorFlow object detection API. For proof of concept, the solution was limited to four clubs:
1. Real Madrid
2. Manchester United
3. Chelsea
4. Barcelona

The process was initiated with preparing data. Images from each club were googled and downloaded in batches using Fatkun batch downlading extension by Chrome. Then they were manually labelled in PASCAL VOC format using Labelimg graphical image annotation tool:
![image](https://user-images.githubusercontent.com/64746481/119837380-f7f54280-bf1b-11eb-91f3-6275741b2943.png)
Total of 370 images were annotated and 90 percent of them were used for training while the other 10 percent were reserved for validation. All images had to be converted to tf_record format to be processed by the API. I firstly converted data to csv and then to tf_record format. The following process consisted of:
1. setting up the environment
2. configuration for training
3. training 
4. inference and testing
Process of each portion is explained with code in the colab notebook: football_logo_detection_CP.ipynb.

Training of the model for only 1000 steps gave good results, some of which are also shown below:
![4](https://user-images.githubusercontent.com/64746481/119848527-5246d100-bf25-11eb-99b1-b2293bca1c55.png) ![4](https://user-images.githubusercontent.com/64746481/119848566-596ddf00-bf25-11eb-9c2c-f976976077f1.png)



