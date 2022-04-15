# csi4106_project

goal is to classify certificates of accreditation for veterinary practices based on country and classification issuing institution

classes are named following this format: country_state/province_certifier (for example canada_ontario_cvo)

## v0.1

simple cnn
one class (canada_ontario_cvo) with very low number of images that are very similar to each other
=> algorithm recognises anything as canada_ontario_cvo because a single class was added

## v0.2

added a "none" class with a large number of random images. the purpose of this class is to serve as anything that is not a certificate of accreditation for veterinary practices (so anything that does not belong to one of the certification classes should be predicted as member of this class)
=> now everything is predicted as a member of the "none" class. This is because of the imbalance between data

## v0.3

Added class weights to solve the issue with imbalanced data
=> improved prediction and properly recognized none vs canada_ontario_cvo images used for training. however, as the images used for training are very similar, there was overfitting and testing with other canada_ontario_cvo images resulted in wrong predictions

## v0.4

Added more variations of the canada_ontario_cvo images to reduce overfitting
Added image rescaling and resizing and augmentation
=> improved accuracy and prediction
==> but now issue is that we getting 100% accuracy, which prolly means overfitting

## v0.5

added tensorboard and models/logs logging

### stuff to do

- save model to folder
- add tf board
- if possible, add other classes
- if possible, try AlexNet
- grayscale?
