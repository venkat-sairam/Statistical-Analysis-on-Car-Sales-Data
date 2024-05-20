
## Statistical Analysis on Car Sales Data

### Why am I interested in this problem?
I am interested in understanding the trends and patterns in car sales which can help in making informed business decisions regarding marketing strategies, inventory management, and customer segmentation.

### Relevant background information for readers:
This data provides insights into car sales across various regions, customer demographics, car preferences, and pricing. It can help in analyzing market demands and customer preferences.

### Prior research on this topic:
Consumer behavior in car purchases, effects of demographics on car choice, and regional sales performance. This provides a comparative analysis to understand any deviations or trends in the car sales data.

### Data Source:
The dataset originates from a collection showcased at a LinkedIn event targeted at students, designed to offer practical insights into real-world data analysis. It was later revealed that the actual source of the data is a Kaggle dataset titled "Car Sales Report," available for public access.

 This dataset aggregates transactions from various dealerships, capturing detailed sales records including buyer demographics, car specifications, and transaction details. The collection method involved aggregating these details from multiple car dealers, making it a rich source for observational study in car sales trends without experimental intervention.

### Questions of interest that I hope to answer:

* Do the mean prices and mean annual incomes significantly differ from their hypothesized values?
* Is there a significant difference in the proportion of SUVs produced by car manufacturers between the years 2020 and 2021?
* Is there a significant difference in car prices between manual and automatic transmissions across the years 2020 and 2021?


## Methods and Results

### Pre-processing:

To prepare the car sales dataset for exploratory data analysis (EDA), several preprocessing steps are essential to ensure the data is clean, accurate, and suitable for analysis. Here’s a detailed description of the EDA methods used before analyis:

* **Identifying NULL values:**

   Identified missing values in each column of the dataset.
  ```r
  missing_values <- sapply(car_sales_data, function(x) sum(is.na(x)))
```

| Column        | MissingCount |
|---------------|--------------|
| date          | 0            |
| gender        | 0            |
| annual_income | 0            |
| dealer_name   | 0            |
| company       | 0            |
| model         | 0            |
| engine        | 0            |
| transmission  | 0            |
| color         | 0            |
| price         | 0            |
| dealer_no     | 0            |
| body_style    | 0            |
| dealer_region | 0            |


* **Duplicate values:**  Removed duplicates to ensure each data point is unique for the analysis.

  ```r
  duplicates <- sum(duplicated(car_sales_data))

  ````

* Data Type Checking: Ensured that numerical and categorical columns are type checked and corrected.

  ```r
  data_types <- sapply(car_sales_data, class)
  data_types_df <- data.frame(
                  Column = names(data_types),
                  data_type = data_types, stringsAsFactors = FALSE)
  ```

* Removed irrelevant features such as `car_id`, `phone`, and `customer_name` as they are almost static and no interesting patterns can be found using these features.

* **Summary Statistics:**
 Generated descriptive statistics that summarize central tendency, dispersion, and shape of the dataset’s numerical attributes.

  ```r
  summary_statistics_numeric <- car_sales_data %>%
                                select_if(is.numeric) %>%
                                summary()
  ```

|             | price | annual_income |
|-------------|-------|---------------|
| Min.        | 1200  | 10080         |
| 1st Qu.     | 18001 | 386000        |
| Median      | 23000 | 735000        |
| Mean        | 28090 | 830840        |
| 3rd Qu.     | 34000 | 1175750       |
| Max.        | 85800 | 11200000      |


* Visualized the Bar charts for gender and transmission featues.
* Scatter plots to check for relationships and correlations between price and annual income features.
* Box plots for gender to visualize statistical summaries and identify outliers.


**The Shapiro-Wilk analysis requires data records to be in the range of `3 <= number of records < 5000`. Therefore, based on this requirement, we created a new sample of size 20% from the original population using stratified sampling technique.**

We can perform Hypothesis testing, bootstrapping, permutation tests, and regression modeling to answer the questions of interest.

### Analysis

#### Bootstrapping

Bootstrap resampling involves repeatedly sampling with replacement from the original data to estimate the sampling distribution of a statistic.
Here, we estimated the mean of the price and annual incomes of the car sales data using bootstrapping.

Histograms are created to visualize the distribution of the scaled bootstrap means for both variables.

Additionally, Q-Q plots are generated to assess whether the scaled bootstrap means follow a normal distribution.


**Interpretation:**

The graphs confirmed that the bootstrapped samples from price and annual incomes were almost normally distributed. This is additionally confirmed by the qq-plots where some points at the right tail were deviated from the reference line.

#### Confidence Intervals for Bootstrapped data

| Variable | Lower CI | Mean Value | Upper CI |
|----------|----------|------------|----------|
| Price    | 27896.40 | 28088.00   | 28279.62 |
| Income   | 821611.5 | 830600.66  | 839589.8 |

#### Hypothesis Test on Bootstrapped data:

* Null Hypothesis:The mean price or mean annual income equals the hypothesized value.

* Alternative Hypothesis:The mean price or mean annual income does not equals the hypothesized value.

#### Hypothesis Test Results/ Analysis:


| Variable | Hypothesized Mean | Confidence Interval | Within CI | T-Statistic | P-Value |
|----------|-------------------|---------------------|------------|-------------|---------|
| Price    | 28370.94          | [27896.40, 28279.62]| FALSE      | -91.52233   | 0       |
| Income   | 842311.3          | [821611.5, 839589.8] | FALSE      | -80.74525   | 0       |



1. Price:
  * The hypothesized mean price is $28370.94.
  * The observed price falls outside the confidence interval limits.
  * The t-statistic value of -91.52233 suggests a significant deviation from the hypothesized mean.
  * A p-value of 0, indicates strong evidence to `reject the null hypothesis`.

2. Income:
  * The hypothesized mean income is $842311.3.
  * Similar to price, the observed income falls outside the confidence interval limits.
  * The t-statistic value of -80.74525 suggests a significant deviation from the hypothesized mean.
  * A p-value of 0, indicates strong evidence `against the null hypothesis`.

####  Normality check on Bootstrapped data:

|   R | P_Value  |
|-----|----------|
| 500 | 0.4454739|
|1000 | 0.6276020|
|2000 | 0.2422149|
|4000 | 0.2963930|
|4999 | 0.1103231|

Since, all the samples have a p-value greater than the threshold of 0.05, we can conclude that the boostrapped data is following normality.

#### ` Two - tailed T-test`

**Null Hypothesis (H0):** The proportion of SUVs produced by car manufacturers in 2020 is the same as the proportion produced in 2021.

**Alternative Hypothesis (H1):** The proportion of SUVs produced by car manufacturers in 2020 is different from the proportion produced in 2021, indicating a shift in consumer demand towards larger vehicles.

  **Statistical method used: two tailed T-test**

Assumptions: Data Should follow normality.

* Shapiro-wilk Test Used to check normality

```r
# Calculating the differences between the two years for each company
suv_proportions_wide$differences <- suv_proportions_wide$`2021` - suv_proportions_wide$`2020`
# Performing the Shapiro-Wilk test for normality on these differences
normality_test_results <- shapiro.test(suv_proportions_wide$differences)
```
Since the w value from the test results is close to 1, we can say that the data set is following normality.

| Test            | Value    |
|-----------------|----------|
| Test statistic - W value | 0.94087  |
| p-value         | 0.1409   |

**Interpretation of Shapiro-wilk Test:**

p-value is greater than the threshold of 0.05 and w-statistic is close to 1. So, we can conclude that the data is following normality.

#### Code Snippet for Two-tailed t-test

  ```r
  suv_proportions_clean <- na.omit(suv_proportions_wide)
  differences <-suv_proportions_clean$`2021`  - suv_proportions_clean$`2020`
  mean_diff <- mean(differences)
  sd_diff <- sd(differences)

  n <- length(differences)

  t_value <- mean_diff / (sd_diff / sqrt(n))
  df <- n-1
  p_value <- 2 * pt(-abs(t_value), df)  # two-tailed test

  ```
  
#### Result:

| Measurement                      | Value      |
|----------------------------------|------------|
| Mean of differences              | 0.02597654 |
| Standard deviation of differences| 0.08213114 |
| t-Value                          | 1.612724   |
| Degrees of freedom               | 25         |
| p-Value                          | 0.1193583  |


#### Two-tailed t-test Analysis:

* Mean difference is positive. So, the number of SUV's made in 2021 are more than 2020.

* A p-value $= 0.1193583$. So, there is $11.9\%$ chance of finding difference in proportions.
  
* The 95% confidence interval $-0.05915002 \text{ to } 0.007196951$. So, we don't have enough evidence to reject the NULL hypothesis.


#### `Hypothesis Testing with Randomization`

* Null Hypothesis (H0): There is no difference in car prices between manual and automatic transmissions.

* Alternative Hypothesis (H1): There is a difference in car prices between manual and automatic transmissions.

#### Analysis

| Year | Observed Difference | p-value | Significance Level | Conclusion |
|------|---------------------|---------|--------------------|------------|
| 2020 | `$2034.726`             | 0.002   | 0.05               | reject null hypothesis |
| 2021 | `$630.0765`          | 0.2808  | 0.05               | No evidence to reject null hypothesis |


For year 2020: There is evidence of a pricing difference between transmission types, so we can reject the NULL hypothesis.

For year 2021, there is no enough evidence to reject the null hypothesis.

#### Confidence Intervals for Hypothesis with Randomization Test


|     Year 2020        | Value                |
|-------------|----------------------|
| Observed Diff | -2034.72  |
| Confidence Interval (2.5%) | -1212.71 |
| Confidence Interval (97.5%) | 1237.54  |

|    Year 2021         | Value                |
|-------------|----------------------|
| Observed Diff | 630.08  |
| Confidence Interval (2.5%) | -1150.24  |
| Confidence Interval (97.5%) | 1132.27|


#### **Regression Modeling**

Sampled dataset is divided into train/test datasets with 80%, 20% size respectively.

```r
# Creating training and testing datasets
train_data <- strat_sample[train_indices, ]
test_data <- strat_sample[-train_indices, ]

```

Simple Linear Regression model is created between  price and annual income and then calculated the fitted values and residuals using pearson's method.

```r
model <- lm(price ~ annual_income, data = train_data)
```




#### Simple Linear Regression Analysis:

* **Linearity**:

most of the points are closeto/along the line of reference. so, linearity exists.

* **Homoscedasticity**: The datapoints are not having constant variance. So homoscedasticity fails with mlr model.

* **Outliers**: Outliers exists mostly at the extremes.

* **Normality**:

There are some points deviating from the red line, particularly at the tails of the graph. This indicates that there may be some skewness in the dataset.So, model is not following the normality principle.

#### Analysis after applying log transformation:

* **Linearity**: Not violated

* **Homoscedasticity**: variance is constant with square root transformation

* Outliers: potential outliers exists even after applying transformation.


* Normality: Not met even after applying square root transformation.


#### Simple Linear Regression Predictions:


| Metric                    | Value                 |
|---------------------------|-----------------------|
| Training RMSE             | 0.471    |
| Test RMSE                 | 0.488    |
| Training R-Squared        | 0.00002 |
| Training Adjusted R-Squared | -0.00023 |

By considering the predictions, we can conclude that the model is weak in terms of predicting the target variable even after applying square root transformations.

#### **Multiple Linear Regression**

created dummy variables for all the categorical variables and omitting the intercept.

```r
dummy_variables_matrix <- model.matrix(~ gender + company +
                          model + engine + transmission + color +
                          body_style + dealer_region - 1, data=strat_sample)
```
After encoding the categorical variables, multiple linear regression model is created between all the features as predictors and price as outcome variable.

```r
lm_dummies = lm(price ~ . ,  data = train_data)
```
once the model is fitted, then plotted graphs against

 * Linearity check: observed values vs predicted variables
 
 * Constant variance check: yhat and r-square.
 
 * Normality check: Q-Q plot

#### Observations:

* Linearity is violated.

* Variance is not constant across the plotted residuals. So, heteroscedasticity exists.

* Normality: Most of the points appear to follow the red line closely from -2 to 2. So, the model is not fully follwing normal distribution. Only a subset of the datapoints are following normality principle.

#### Analysis of MLR with Transformations:

* **Linearity:** MLR follows linearity with/without applying transformations.

* **Shapiro -wilk Test**: p-value < 2.2e-16. So, the error terms are not following normality even after applying transformations.

* **Durbin-Watson Test**: DW Statistic is 2.0013 and p-value is greater than the threshold value of 0.05. So, we can conclude that there is no auto-correlation exists in the MLR model.

* **Residuals vs Leverage** Check:
The points "3762", "2442" exceed certain thresholds in terms of leverage or residual and observed a few points with higher leverage.


#### **Model Selection:**


Performed backward elimination for the previously created regression models.This method iteratively removes variables with the highest p-values until all variables have p-values below a significance threshold.

It calculates the Mean Squared Prediction Error (MSPE) for each updated model, storing the values for comparison. If the MSPE decreases with variable removal, the model is updated. The process continues until no further improvement is observed.

Finally, prints the MSPE values collected after each model update.

#### Best Model Analysis


**Residuals vs. Fitted:**

* **Expectation**: The points should be randomly dispersed with no clear patterns.

* **Observation**: There appears to be a curve in the pattern of points, suggesting the possibility of a non-linear relationship.

Normal Q-Q (Quantile-Quantile):

* **Expectation**: Points should lie on the reference line if the residuals are normally distributed.

* **Observation**: While most points align, some at the right ends deviate. This indicates that the right tails of the distribution are deviating from the line and are not followig the normality.

Scale-Location (Spread vs. Fitted Values):

* **Expectation**: The spread of residuals should be consistent across the range of fitted values.

* **Observation**:  The spread of residuals is fairly constant across the range of fitted values. This suggests that the model follows homoscedasticity.

Residuals vs. Leverage:

* **Expectation**:

  * The x-axis represents "Leverage". Observations near the mean of the predictors have low leverage, while observations far from the mean have high leverage.
  
  * The y-axis represents "Standardized residuals," indicating the deviation of the observed values from the values predicted by the model, scaled in terms of the standard error.
  
  * The horizontal dotted lines represent reference lines for Cook's distance.This helps us to  identify potentially influential points.
  
  * The red line is typically a smooth fit to the data. A horizontal line with no pattern suggests that residuals are randomly distributed, not influenced by leverage.

* **Observation**:

  * Points with low leverage clustered around the mean of the predictors.
  
  * Observed a few points with higher leverage. These are the points that can potentially influence the regression line more than others.
  
  * The points that are far from zero on the y-axis are the ones with large residuals — these could be outliers or points that the model does not predict well.
  
  * The points "3463", "476" exceed certain thresholds in terms of leverage or residual size, or both are highlighted by the plotting function.


#### Finiding the index of model with lowest MSPE:

```r
# Find the index of the best model with the lowest MSPE
bestMSPE_Index <- which.min(mspe_values_after_each_step)
# Retrieve the best MSPE value using the index
bestMSPE <- mspe_values_after_each_step[bestMSPE_Index]

cat("Using MSPE as a criterion, best model is model- ", bestMSPE_Index,
    "with MSPE Value equal to ", bestMSPE, "\n")
```

**Observation**: Using MSPE as a criterion, best model is model-  1 with MSPE Value equal to  1013945382

#### AIC/BIC/R-Squared:

Iteratively removed selected features from the fitted model and then updated the model with revised feature set to compute the AIC, BIC and r-squared values for the revised model.

#### Best AIC/BIC/R-squared values

| Model Selection          |  Value |
|--------------------------|------------|
| Best Model by AIC        | 38355.93   |
| Best Model by BIC        | 39555.85   |
| Best Model by Adjusted R-squared | 0.235483 |

## Conclusion:

### Conclusions and Assumptions:

Through this analysis, we concluded that the linear regression model is functional with some limitations due to violation of some key assumptions.

* **Residual Analysis**: The deviations observed in the Q-Q plot suggest challenges with normality.

* **Outlier Analysis**: The presence of outliers and some influential points might be the reason for the model to predict weakly.

* **Predictive Accuracy**: The RMSE values for both training and test datasets show reasonable error margins, but the low R-squared values explains about the model's explanatory power and its generalizability to other data sets.

From this analysis, learned the importance of thoroughly testing statistical assumptions and exploring data characteristics before finalizing a model. These steps are crucial for understanding the limitations and potential biases in model predictions.

### Ideas for extending the research:

**Exploring Additional Data Transformations**: Given the issues with non-normal residuals and heteroscedasticity, applying different transformations or adopting non-linear models could help in achieving better model fit and prediction accuracy.

**Advanced Modeling Techniques:** Exploring machine learning methods like random forests or gradient boosting machines, which can handle non-linearity and complex interactions more effectively than linear regression, might provide better predictive performance.

