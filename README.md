
# 02582 Computational Data Analysis `Case 2`

- [02582 Computational Data Analysis `Case 2`](#02582-computational-data-analysis-case-2)
  - [Dimensionality Reduction](#dimensionality-reduction)
  - [Cluster Analysis](#cluster-analysis)
  - [Feature Analysis](#feature-analysis)
  - [Tasks](#tasks)


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


## Tasks


### Spaces considered
- [ ] Find latent space $\mathcal{S}_{AE}$ with auto encoder
- [ ] Find gray scale space $\mathcal{S}_{gray}$ by simply grayscaling the images
- [ ] Consider other latent spaces

### Clustering
- [ ] Find clusters $\mathcal{C}_{AE}$ in latent space $\mathcal{S}_{AE}$ via k-means
- [ ] Find clusters $\mathcal{C}_{gray}$ in gray scale space $\mathcal{S}_{gray}$ via k-means
- [ ] Experiment with number of clusters and other cluster parameters

### Performance measures

- [ ] Find performance measure $P$ to use for identifying purity of found clusters
- [ ] Compare performance on spaces $\mathcal{S}_{AE}$ and $\mathcal{S}_{gray}$

### Visualization
- [ ] Visualize performance of AE, that is reconstruction image, train/test loss etc.
- [ ] Visualize clusters in their space $\mathcal{S}_{AE}$ and $\mathcal{S}_{gray}$
- [ ] Visualize purity of clusters in their spaces $\mathcal{S}_{AE}$ and $\mathcal{S}_{gray}$

### Additional (if time allows)
- [ ] Visualize how latent features impact the clustering (e.g. by showing how a single image is mapped to different clusters for different values of the latent feature)