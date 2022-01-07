# Explainable Artificial Intelligence for Authorship Attribution on Social Media

This page contains supplementary materials of our submitted ICASSP'22 paper (under revision): "_Explainable Artificial Intelligence for Authorship Attribution on Social Media_".


## Section 4.4. Coverage of Explanations

Percentage of the most relevant character 4-grams that contain elements missed by unigram LIME (e.g., space, punctuation, and emojis). Character 4-grams capture these elements, and their writing patterns are essential to attribute authorship. However, the original LIME is unable to identify the majority of them, generating poor explanations in the case of authorship attribution of small messages.

<figure>
  <img src="./strong_tps_non_alpha_ratio_by_author.png" alt=""/>
  <figcaption>Percentage of most relevant character 4-grams that contain non-alphanumeric characters (by author in the CONF dataset).</figcaption>
</figure>

<br>
<br>

<figure>
  <img src="./strong_tps_non_alpha_ratio_overall.png" alt=""/>
  <figcaption>Percentage of most relevant character 4-grams that contain non-alphanumeric characters (overall in the CONF dataset).</figcaption>
</figure>

<br>
<br>

<figure>
  <img src="./random_non_alpha_ratio_by_author.png" alt=""/>
  <figcaption>Percentage of most relevant character 4-grams that contain non-alphanumeric characters (by author in the RAND dataset).</figcaption>
</figure>

<br>
<br>

<figure>
  <img src="./random_non_alpha_ratio_overall.png" alt=""/>
  <figcaption>Percentage of most relevant character 4-grams that contain non-alphanumeric characters (overall in the RAND dataset).</figcaption>
</figure>

