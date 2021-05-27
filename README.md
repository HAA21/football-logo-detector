# football-logo-detector

I have implemented football club logo detector using TensorFlow object detection API. For proof of concept, the solution was limited to four clubs:
1. Real Madrid
2. Manchester United
3. Chelsea
4. Barcelona

The process was initiated with preparing data. Images from each club were googled and downloaded in batches using Fatkun batch downlading extension by Chrome. Then they were manually labelled in PASCAL VOC format using Labelimg graphical image annotation tool:
![image](https://user-images.githubusercontent.com/64746481/119837380-f7f54280-bf1b-11eb-91f3-6275741b2943.png)
Total of 370 images were annotated and 90 percent of them were used for training while the other 10 percent were reserved for validation. All images had to be converted to tf_record format to be processed by the API. I firstly converted data to csv and then to tf_record format.  

