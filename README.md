### CSE150A Project Milestone 4
Dataset: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset/data

## Describe your dataset (4pts):
Our dataset is a list of movies with titles and genres with ratings by 138493 users between January 09, 1995 and March 31, 2015. We want to recommend to the user the best movie(s) given what movies the user has already watched. The data is relevant to probabilistic modeling because the user inputs are undetermined, and we can use the random user inputs to predict the next result. Our data will be preprocessed by parsing the movies into a dictionary, where keys are genres and values are arrays of movies with that genre. Also filter out movies with no genres.
The dataset has 27.3k entries and is reasonably processable. The dataset we are using is from MovieLens. It is randomly sampled and reliable. The data is categorical.
“The datasets describe ratings and free-text tagging activities from MovieLens, a movie recommendation service. It contains 20000263 ratings and 465564 tag applications across 27278 movies. These data were created by 138493 users between January 09, 1995 and March 31, 2015. This dataset was generated on October 17, 2016. Users were selected at random for inclusion. All selected users had rated at least 20 movies.”
	
## Project Proposal:
Our model should find the best movie recommendations based on the movies recently watched by the user. Uncertainty modeling is important because recommending movies isn’t objective and is subjective to what the user values. In this context, non-probabilistic approaches don’t take into account how users often choose movies with little to no relation to movies previously watched.

## Agent in terms of PEAS:
Performance measure: Accuracy depending on user, recall, and precision
Environment: Terminal or code editor
Actuators: Screen/text output
Sensors: User input / list of movies watched

## Methodology:
We are using a Naïve Bayes model. We will use maximum likelihood to calculate CPT values, since we are using a naive bayes model. We will assume that movies with no genres won’t be chosen and will never be recommended. We will also assume that there will be an optimal recommendation. To evaluate our model, we will sample 80% of a user's ratings and movies watched, and 

1. Prior probability of liked \(y\):

$$
P(y) = \frac{\text{Number of movies liked } y}{\text{Total number of movies rated}}
$$

2. Conditional probability of features \(X\) given liked \(y\) (assuming feature independence):

$$
P(X \mid y) = \prod_{i=1}^n P(x_i \mid y)
$$

where

$$
P(x_i \mid y) = \frac{1 + \text{count of movies where } x_i \text{ is present and liked } y}{1 + \text{total count of movies liked } y}
$$

*(Laplace smoothing is applied to avoid zero probabilities)*

3. Posterior probability of liked \(y\) given features \(X\):

$$
P(y \mid X) \propto P(y) \times \prod_{i=1}^n P(x_i \mid y)
$$

4. Prediction rule:

$$
\hat{y} = \arg\max_{y \in \{0,1\}} P(y \mid X)
$$
