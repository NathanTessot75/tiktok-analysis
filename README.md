# Can Machine Learning Predict If Your TikTok Will Be a Hit or a Flop?

**Predicting the number of likes of a TikTok video from its metadata**

Nathan Tessot · CPES Data Science & Cultural Sciences, Université PSL · January 2026

📄 **[Read the report (PDF)](report/TikTok_Likes_Prediction_Tessot.pdf)** · 📓 [Notebook](tiktok_likes_prediction.ipynb)

---

## Question

Can the number of likes of a TikTok video be predicted from its characteristics
(views, shares, comments, downloads, duration, transcription) and from the status
of its author (verified, banned)? Since the target is continuous, this is a
regression problem.

## Data and method

- **Data**: a public Kaggle dataset of 19,382 TikTok videos labelled as claims or
  opinions, 19,084 after removing missing values, described in
  [`data/codebook.xlsx`](data/codebook.xlsx).
- **Feature engineering**: 22 new variables (27 predictors in total), including log
  transforms, engagement ratios, encoded categorical variables, text features from
  the transcription and duration-based rates.
- **Protocol**: standardized features, 75/25 train/test split, three models tuned
  with 5-fold cross-validation and compared on RMSE:
  - linear regression
  - LASSO regression (regularization strength tuned over seven values from 1 to 10⁶)
  - neural network (multilayer perceptron; architecture, activation, solver,
    learning rate and batch size tuned over 120 combinations)

## Results

| Model | Train RMSE | Test RMSE | Test R² |
|-------|-----------:|----------:|--------:|
| Linear regression | 49,436 | 49,927 | 0.864 |
| LASSO regression | 49,882 | 50,410 | 0.861 |
| **Neural network** (2 hidden layers, 8 and 4 units, ReLU) | **44,611** | **44,533** | **0.891** |

![Predicted vs. actual likes for the three models](report/figures/predicted_vs_actual.png)

The neural network performs best, with a test error about 11% lower than the linear
models. Its test RMSE (about 44,500 likes) is roughly a third of the standard
deviation of likes, and train and test errors are close, so the model does not
overfit. The report discusses two limits: the RMSE is dominated by the most popular
videos (the median video has about 3,400 likes), and the strongest predictors
(views, shares, comments) are measured at the same time as the likes.

## Repository structure

```
├── tiktok_likes_prediction.ipynb   Full analysis: cleaning, features, models, evaluation
├── data/
│   ├── tiktok_dataset.csv          Raw dataset
│   └── codebook.xlsx               Description of each variable
├── report/
│   ├── TikTok_Likes_Prediction_Tessot.pdf
│   ├── main.tex                    LaTeX source of the report
│   ├── figures/
│   └── scripts/reproduce_models.py Refits the selected models and redraws the figures
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook tiktok_likes_prediction.ipynb
```

The full neural-network grid search takes several minutes.

To regenerate the report figures from the selected hyperparameters (about two
minutes, no grid search):

```bash
cd report/scripts
python reproduce_models.py
```
