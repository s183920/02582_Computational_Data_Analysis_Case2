# 02582 Computational Data Analysis `Case 2`

In this project an exploratory analysis of the UTKFaces dataset will be carried out using unsupervised learning.

The data contains over 20,000 RGB images of faces and labels describing age, gender and race. 


## Dimensionality Reduction

The images are alinged and cropped into a shape of (200 x 200 x 3) pixels each. Analysing the images directly is deemed infeasible as each contains 120,000 features and, therefore, it is desired to reduce the dimensionality of the data.

This is done using an Variational AutoEncoder (VAE).

The VAE will be compared to PCA...


## Cluster Analysis
In the latent space encoded by the VAE we will carry out our analysis and explore how much information remains in the encoded data.

Through clustering the latent space will be investigated - how is data grouped? Which observation are closely related? Can the observations be divided into race, gender, age? 

## Feature Analysis
The VAE aims to structure the latent space to be gaussian, making it possible to apply small changes to a latent observation without changing the decoded result too much. 

We will try to investigate which directions in the latent space describes, age, race or gender by decomposing the latent space using some method (PCA, NNMF, ICA ??) and investigate how the components are related to a specific character traits. 
What do theses features look like in the real space (decoded)? What happens if we modify the observations in the latent space and decode them? 
Can we change the gender/race of a person? 
