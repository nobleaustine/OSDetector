# OSDetector
Enhanced Oil 🛢️ Spill Detection and Management in SAR 📡 Imagery 🖼️ Through Deep Learning 🖼️ and Contextual Data 📂 Integration 

![alt text](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT-VnaUuciJoICGHqwY4VyrN_1IGsLP5WROnw&s)

Abstract:

Oil spills are a significant environmental hazard, causing extensive damage to marine ecosystems and impacting global fisheries. Timely and accurate detection of oil spills is crucial for minimising their environmental impact and enabling effective response efforts. Synthetic Aperture Radar (SAR) imagery offers a powerful solution for oil spill detection due to its capability to operate under various weather conditions and through cloud cover. Nonetheless, SAR imagery encounters challenges such as speckle noise, varying incidence angles, and interference from landmasses. To overcome these challenges, our approach incorporates a comprehensive preprocessing pipeline designed to enhance SAR image quality. This pipeline addresses speckle noise reduction, incidence angle correction, land masking, and data normalisation, significantly improving detection accuracy. The framework further integrates additional contextual information, including wind speed, ship routes, and oil pipeline locations, to refine detection performance. A Bayesian theory-based image filtering technique is employed to assess the probability of oil spills in specific areas. For feature extraction, we utilise a VGG-16-based Convolutional Neural Network (CNN), complemented by a Region Proposal Network (RPN) to generate candidate bounding boxes for potential oil spills. These candidates are then refined using Non-Maximum Suppression (NMS) to ensure precise localization. Additionally, instead of relying on a single model, our approach plans to implement one or more models to further enhance detection outcomes. Overall, this advanced framework represents a significant improvement in the detection and management of marine oil spills, leveraging deep learning techniques and contextual data to enhance accuracy and facilitate prompt response. By addressing key challenges in SAR imagery and integrating additional information, this approach offers a comprehensive solution for managing the environmental impacts of oil spills.

The abstract include : 
* Why Oil spill detection?? 🛢️
* Its harmful effects 💀
* SAR dataset 📂 - Synthetic Aperture Radar imagery is a powerful tool 📶
* Its advantages
* Additional approaches
* Filtering the SAR 📡 dataset based on the probability 📈 ( Bayesian theory ) of oil spill in an area using ship 🚢 route data, oil pipeline locations etc.
   * Most probable source 🧮
   * Alerts and Solutions ⚠️
* Deep-learning architectures 🧠
* VGG-16-based Convolutional Neural Network 🤖 (CNN) for feature extraction 
* Region Proposal Network (RPN) for generating candidate bounding boxes
* Refined using Non-Maximum Suppression (NMS) 🧮

New Methods Implemented
* Instead of one model many models
* Dataset from different source 
* SAR 📡 dataset
* Ship route data 🚢

Methodology
* Get a model which detects the oil spill from SAR 📡 data 📂
* Add a predicting layer/function which filter out the area/SAR images which most likely will never have a oil spill
* Thus increasing the model accuracy 📈
* Stressing on the fact that our model uses many data 📂 sources to avoid the unlikely areas/SAR 📡 images which need not be put through the model.

Origin of oil spill : 
* Rather than finding the oil spill in the satellite image 🖼️ we find the origin of the oil spill. Using the dataset containing flow of the oil spill, wind speed etc.

Probability of oil spill : 
* Filtering out the dataset prior to the oil spill detection by probability factor. Using a dataset containing information about ship routes 🚢 , the latitude and longitude information 🌐 etc. This tells us where the ship went to, the area to search for oil spill is reduced. This is the density map of the ship routes 🚢 . To obtain ship route latitude and longitude 🌐 data from MarineTraffic, you generally have two main options:

