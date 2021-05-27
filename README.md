# football-logo-detector

I have implemented football club logo detector using TensorFlow object detection API. For proof of concept, the solution was limited to four clubs:
1. Real Madrid
2. Manchester United
3. Chelsea
4. Barcelona

The process was initiated with preparing data. Images from each club were googled and downloaded in batches using Fatkun batch downlading extension by Chrome. Then they were manually labelled in PASCAL VOC format using Labelimg graphical image annotation tool:
![image](https://user-images.githubusercontent.com/64746481/119837380-f7f54280-bf1b-11eb-91f3-6275741b2943.png)
Total of 370 images were annotated and 90 percent of them were used for training while the other 10 percent were reserved for validation. All images had to be converted to tf_record format to be processed by the API. I first converted data to csv and then to tf_record format. The following process consisted of:
1. setting up the environment
2. configuration for training
3. training 
4. inference and testing

Process of each portion is explained with code in the colab notebook: football_logo_detection_CP.ipynb. Training of the model for only 1000 steps gave good results, some of which are also shown below. This proves that by training the model for more steps and on data from other clubs, we can have a fully functional football club logo detector.

![12](https://user-images.githubusercontent.com/64746481/119849170-d6995400-bf25-11eb-94fd-58a62a3f591e.png) ![5](https://user-images.githubusercontent.com/64746481/119850168-b7e78d00-bf26-11eb-9a23-7c7c330da51d.png)
 ![4](https://user-images.githubusercontent.com/64746481/119850042-9b4b5500-bf26-11eb-91da-eeb6cd5fff98.png) ![11](https://user-images.githubusercontent.com/64746481/119850820-5116a380-bf27-11eb-8fa6-907d364be944.png)

 
Resources:
1. https://gilberttanner.com/blog/tensorflow-object-detection-with-tensorflow-2-creating-a-custom-model
2. https://heartbeat.fritz.ai/end-to-end-object-detection-using-efficientdet-on-raspberry-pi-3-part-2-bb5133646630
3. https://towardsdatascience.com/train-an-object-detector-using-tensorflow-2-object-detection-api-in-2021-a4fed450d1b9






