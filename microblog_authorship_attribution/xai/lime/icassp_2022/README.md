# Explainable Artificial Intelligence for Authorship Attribution on Social Media

This page contains supplementary materials of our submitted ICASSP'22 paper (under revision) "_Explainable Artificial Intelligence for Authorship Attribution on Social Media_".

We present results from experiments running predictions from two models for authorship attribution of small messages described below:

1. Theophilo et al. 2021 - Authorship Attribution of Social Media Messages. Theophilo, Antonio and Giot, Romain and Rocha, Anderson. IEEE Transactions on Computational Social Systems. 2021. IEEE.
1. Rocha et al. 2016 - Authorship attribution for social media forensics. Rocha, Anderson and Scheirer, Walter J and Forstall, Christopher W and Cavalcante, Thiago and Theophilo, Antonio and Shen, Bingyu and Carvalho, Ariadne RB and Stamatatos, Efstathios. IEEE transactions on information forensics and security. 2016. IEEE.

## Section 4.1. Dataset Organization

We generated predictions for all samples in the validation set and, from these samples, we defined, or each model, two subsets for which to generate LIME explanations. The first (RAND) has 1,000 messages randomly chosen (20 messages from each one of the 50 authors), and the second (T-CONF) is the set of all correct predictions with high confidence ($>$ 0.9), consisting of a subset of 2977 samples for the model [1] and 451 samples for model [2].

## Section 4.2. Interpretation Evaluation

Here we present a comparison of the explanations offered by the standard unigram LIME method and our proposed LIME extension using character n-grams.


<figure>
  <img src="./example_1_original.png" alt=""/>
</figure>
<figure>
  <img src="./example_1_extended.png" alt=""/>
  <figcaption>Example 1 (model [1]).</figcaption>
</figure>

<br>
<br>


<figure>
  <img src="./example_2_original.png" alt=""/>
</figure>
<figure>
  <img src="./example_2_extended.png" alt=""/>
  <figcaption>Example 2 (model [1]).</figcaption>
</figure>

<br>
<br>


<figure>
  <img src="./example_3_original.png" alt=""/>
</figure>
<figure>
  <img src="./example_3_extended.png" alt=""/>
  <figcaption>Example 3 (model [1]).</figcaption>
</figure>

<br>
<br>


# **TODO: figures from Fernanda **


## Section 4.3. Redundancy of Perturbed Samples

| **Representation** | **RAND**                | **T-CONF**              |
| :---               |          :---:          |          :---:          |
|                    | Mean+-Std (Max)         | Mean+-Std (Max)         |
| unigram            | 55.19 +- 19.74 (89.09)  | 51.62 +- 22.06 (91.47)  |
| **char-4-gram**    | **9.84 +-7.98 (29.73)** | **7.96+- 6.27 (22.78)** |


## Section 4.4. Coverage of Explanations

Percentage of the most relevant character 4-grams that contain elements missed by unigram LIME (e.g., space, punctuation, and emojis). Character 4-grams capture these elements, and their writing patterns are essential to attribute authorship. However, the original LIME is unable to identify the majority of them, generating poor explanations in the case of authorship attribution of small messages.

<figure>
  <img src="./strong_tps_non_alpha_ratio_by_author.png" alt=""/>
  <figcaption>Percentage of most relevant character 4-grams that contain non-alphanumeric characters (by author in the CONF dataset - model [1]).</figcaption>
</figure>

<br>
<br>

<figure>
  <img src="./strong_tps_non_alpha_ratio_overall.png" alt=""/>
  <figcaption>Percentage of most relevant character 4-grams that contain non-alphanumeric characters (overall in the CONF dataset - model [1]).</figcaption>
</figure>

<br>
<br>

<figure>
  <img src="./random_non_alpha_ratio_by_author.png" alt=""/>
  <figcaption>Percentage of most relevant character 4-grams that contain non-alphanumeric characters (by author in the RAND dataset - model [1]).</figcaption>
</figure>

<br>
<br>

<figure>
  <img src="./random_non_alpha_ratio_overall.png" alt=""/>
  <figcaption>Percentage of most relevant character 4-grams that contain non-alphanumeric characters (overall in the RAND dataset - model [1]).</figcaption>
</figure>

