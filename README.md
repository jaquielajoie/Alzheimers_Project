# Models made using PITT Corpus 
https://dementia.talkbank.org/access/English/Pitt.html
(specifically Cookie Theft stimulus photo for the Control group and the Dementia group.)

# Alzheimers_Project
This code is used to transform the PITT data set (dementia) into json files to pipe into a keras model.

The model is 3 convolutional layers with a regressor classifier as output. 

0=Dementia (with my model)
1=Control

THIS CODE NEEDS TO BE REFACTORED. PLEASE DON'T JUDGE IT IS A WORK IN PROGRESS.

# Performance

current results (30 epochs):
LOSS, MAE, MSE
[0.05875355005264282, 0.15116603672504425, 0.05875355005264282]

# End goal

- Put this on a smart phone.
- write an actual README
