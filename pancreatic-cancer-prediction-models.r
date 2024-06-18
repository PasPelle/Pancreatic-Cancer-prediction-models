# Libraries for data manipulation/analysis
library(tidyverse)
library(data.table)
library(patchwork)
library(ggcorrplot)
library(glue)
library(nnet)

# ML libraries
library(randomForest)
library(xgboost)
library(caret)
library(caretEnsemble)
library(e1071)
library(pROC)

# Read the dataset
dt = fread("/kaggle/input/urinary-biomarkers-for-pancreatic-cancer/Debernardi et al 2020 data.csv")
head(dt, 10)

######
# QC #
######

# Replace missing values with NA for numeric columns
dt[, (names(dt)) := lapply(.SD, function(x) ifelse(is.na(x) | x == "", NA, x)), .SDcols = names(dt)]

# Check if missing values are replaced
colSums(is.na(dt))


# Data types and structure
str(dt)

# Convert diagnosis column to categorical
dt$diagnosis <- as.character(dt$diagnosis)

#Shapiro-Wilk normality test
for (name in names(dt)) {
    if (is.numeric(dt[[name]])) {
        result <- shapiro.test(dt[[name]])
        if (result$p.value < 0.05) {
            print(paste(name, "is NOT normally distributed.", " p.value:",
                        format(result$p.value, digits = 4)))
        } else {
            print(paste(name, "is normally distributed."," p.value:",
                        format(result$p.value, digits = 4)))
        }
    }
}


#############################################
# Display distribution of numerical columns #
#############################################

# Filter specific columns
variables_to_keep <- c("age", "plasma_CA19_9", "LYVE1", "REG1A", "TFF1", "creatinine", "REG1B")
dt_filtered <- dt[,..variables_to_keep, with = FALSE]

# Reshape data into long format for ggplot
dt_long <- pivot_longer(dt_filtered, cols = everything(), names_to = "variable", values_to = "value")

# Convert the "value" column to numeric if it's not already numeric
dt_long$value <- as.numeric(dt_long$value)

# Make plots for each variable
# Used lapply to make a list of plots

plots_numeric <- lapply(unique(dt_long$variable), function(var) {
    ggplot(subset(dt_long, variable == var),
          aes(x = value)) +
    geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
    labs(title = paste(var),
        x = var, y = "Frequency") +
    theme_minimal()
})

# Combine plots in one panel
combined_plots <- wrap_plots(plots_numeric, ncol = 3)
print(combined_plots)

# Plot the balance of categorical columns
categorical_columns <- c('sex','diagnosis','sample_origin','patient_cohort')

plots_categorical <- lapply(categorical_columns, function(var) {
    
    # count the occurrencies of each category
    category_count <- table(dt[[var]])

    # convert counts to a data frame for plotting
    category_data <- data.frame(
        category = names(category_count),
        count = as.numeric(category_count))
    
    # bar plot
    ggplot(category_data, aes(x = category, y = count)) +
    geom_bar(stat = "identity", fill = "skyblue", color = "black") +
    labs(title = paste("Frequency of",var), x = "Category", y = "N") +
    theme_minimal()
    
})

# combine plots in one panel
combined_plots <- wrap_plots(plots_categorical, ncol = 2)
print(combined_plots)

########################
# CORRELATION ANALYSIS #
########################

# Define the categorical and numerical columns to include
categorical_columns <- c('sex', 'diagnosis', 'sample_origin', 'patient_cohort')
numeric_column_names <- c("age", "plasma_CA19_9", "creatinine", "LYVE1", "REG1B", "TFF1", "REG1A")

# Subset the data table to get categorical columns
categorical_data_df <- dt[, .SD, .SDcols = categorical_columns]

# Convert all columns to factors to ensure that the categorical data is ready for one-hot encoding
categorical_data_df[] <- lapply(categorical_data_df, as.factor)

# Convert categorical data to model matrix (one-hot encoding)
dt_one_hot <- model.matrix(~ . - 1, data = categorical_data_df)  # '- 1' removes the intercept

# Convert the model matrix to a data frame
dt_one_hot_df <- as.data.frame(dt_one_hot)

# Combine numerical columns with the one-hot encoded categorical columns
numeric_data_df <- dt[, .SD, .SDcols = numeric_column_names]
dt_combined <- cbind(numeric_data_df, dt_one_hot_df)

# Impute missing values with the column median for numerical columns (it's more robust for non-normal distribution)
dt_combined <- dt_combined %>% 
  mutate(across(all_of(numeric_column_names), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Remove columns with zero variance
dt_combined <- dt_combined %>% select(where(~ var(.) > 0))

# Compute Pearson correlation coefficients
correlation_matrix <- cor(dt_combined, use = "pairwise.complete.obs", method = "pearson")

# Plot the correlation matrix
ggcorrplot(correlation_matrix, method = "circle", type = "upper", 
           lab = TRUE, lab_size = 3, colors = c("blue", "white", "red"), 
           title = "Correlation Matrix", hc.order = TRUE)


#######################
# Logistic Regression #
#######################

# Prepare the data:
# Assuming diagnosis1 is the case where both diagnosis2 and diagnosis3 are 0
dt_combined$diagnosis1 <- as.integer(dt_combined$diagnosis2 == 0 & dt_combined$diagnosis3 == 0)

# Define X and y
X <- select(dt_combined, -starts_with("diagnosis"))
y <- apply(dt_combined[, c("diagnosis1", "diagnosis2", "diagnosis3")], 1, which.max) - 1  # adjust index to start from 0

# Split the data into training and testing sets
set.seed(42)
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

# Fit logistic regression model
model <- multinom(as.factor(y_train) ~ ., data = X_train)

# Make predictions
predictions <- predict(model, newdata = X_test)

# Ensure predictions and actual test labels are factors with the same levels
all_levels <- sort(unique(c(y_train, y_test)))  # Get all levels from both training and testing sets

predictions_factor <- factor(predictions, levels = all_levels)
y_test_factor <- factor(y_test, levels = all_levels)

# Evaluate the model
accuracy <- mean(predictions_factor == y_test_factor)
print(glue("Accuracy: {accuracy * 100}%"))

# Detailed performance metrics using confusion matrix
conf_mat <- confusionMatrix(predictions_factor, y_test_factor)
print(conf_mat)

# Additional performance metrics
print(conf_mat$byClass) 

###########
# Xgboost #
###########

# Define X and y
X <- select(dt_combined, -starts_with("diagnosis"))
y <- apply(dt_combined[, c("diagnosis1", "diagnosis2", "diagnosis3")], 1, which.max) - 1

# Split the data into training and testing sets
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.67, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Ensure y_train and y_test are numeric
y_train <- as.numeric(y_train)
y_test <- as.numeric(y_test)

# library(warnings)
options(warn = 1)

# Convert the data to DMatrix format
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

# Hyperparameter tuning
tune_grid <- expand.grid(
  nrounds = c(50, 100, 150),
  eta = c(0.01, 0.1),
  max_depth = c(3, 4),
  gamma = c(0, 1),
  colsample_bytree = c(0.5, 0.7, 1.0),
  min_child_weight = c(1, 3, 5),
  subsample = c(0.5, 0.7, 1.0)
)

# Train control with cross-validation
train_control <- trainControl(method = "cv", number = 5, verboseIter = FALSE)

# Train the XGBoost model with tuning
xgb_tune <- train(
  x = as.matrix(X_train),
  y = y_train,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid,
  objective = "multi:softmax",
  num_class = length(unique(y)), 
  iteration_range = c(1, max(tune_grid$nrounds)),
  verbosity = 0,
  verbose = FALSE
)
           
# Best model
best_model <- xgb_tune$finalModel

# Make predictions
predictions <- predict(best_model, newdata = as.matrix(X_test), iteration_range = c(1, best_model$niter))
predictions <- as.numeric(predictions)

# Convert predictions and y_test to factors with the same levels
predictions <- factor(predictions, levels = 0:(length(unique(y)) - 1))
y_test_factor <- factor(y_test, levels = 0:(length(unique(y)) - 1))

# Evaluate the model
confusion_mtx <- confusionMatrix(predictions, y_test_factor)



# Extract accuracy and other parameters from the confusion matrix
accuracy <- confusion_mtx$overall["Accuracy"]
print(glue("Accuracy: {accuracy * 100}%"))

# Print class-wise metrics
print(confusion_mtx$byClass)

# Extract feature importance
importance_matrix <- xgb.importance(feature_names = colnames(X_train), model = best_model)

# Plot the feature importance
ggplot(importance_matrix, aes(x = reorder(Feature, -Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(x = "Features", y = "Importance", title = "Feature Importance") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

###########################
# Support Vector Machines #
###########################

# Define training control
train_control <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = "final",
  classProbs = FALSE,
  verboseIter = FALSE,
  summaryFunction = defaultSummary
)

# Define the parameter grid for tuning
tune_grid <- expand.grid(
  C = 2^(-5:2),  # Cost parameter
  sigma = 2^(-5:2)  # RBF kernel parameter
)

# Ensure y_train and y_test are factors
y_train <- factor(y_train)
y_test <- factor(y_test)

# Train the SVM model with RBF kernel
svm_model <- train(
  x = X_train,
  y = y_train,
  method = "svmRadial",
  trControl = train_control,
  tuneGrid = tune_grid,
  preProcess = c("center", "scale")
)

# Print the best model's tuning parameters
cat("Best Model Parameters:\n")
print(svm_model$bestTune)

# Make predictions using the correct type
predictions <- predict(svm_model, newdata = X_test, type = "raw")

# Convert predictions to a factor with the same levels as y_train for proper evaluation
predictions_factor <- factor(predictions, levels = levels(factor(y_train)))
y_test_factor <- factor(y_test, levels = levels(factor(y_train)))

# Evaluate the model using the confusion matrix
conf_mat <- confusionMatrix(predictions_factor, y_test_factor)
print(conf_mat)


# Remove diagnosis1 and diagnosis2 columns and use only diagnosis3 as target
dt_combined_binary <- dt_combined %>% select(-diagnosis1, -diagnosis2)

#######################
# Logistic regression #
#######################

# Ensure target is a factor for binary classification
dt_combined_binary$diagnosis3 <- as.factor(dt_combined_binary$diagnosis3)

# Define X and y
X <- select(dt_combined_binary, -diagnosis3)
y <- dt_combined_binary$diagnosis3

# Train-Test split
set.seed(42)
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

# Fit logistic regression model
model <- glm(y_train ~ ., data = X_train, family = binomial)

# Make predictions
predictions <- predict(model, newdata = X_test, type = "response")
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Convert predictions and y_test to factors
predicted_classes <- as.factor(predicted_classes)
y_test_factor <- as.factor(y_test)

# Evaluate the model
accuracy <- mean(predicted_classes == y_test_factor)
print(glue("Accuracy: {accuracy * 100}%"))

# Detailed performance metrics
conf_mat <- confusionMatrix(predicted_classes, y_test_factor)
print(conf_mat)



##############################################
# Ensemble method for multiple binary models #
##############################################

# Suppress xgboost warnings
options(warn = -1)

# Ensure diagnosis3 is a factor for binary classification and rename levels
dt_combined_binary$diagnosis3 <- as.factor(dt_combined_binary$diagnosis3)
levels(dt_combined_binary$diagnosis3) <- make.names(levels(dt_combined_binary$diagnosis3))

# Define X and y
X <- select(dt_combined_binary, -diagnosis3)
y <- dt_combined_binary$diagnosis3

# Split the data into training and testing sets
set.seed(42)
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

# Define train control
train_control <- trainControl(method = "cv", number = 5, savePredictions = "final", classProbs = TRUE, verboseIter = FALSE)

# Train logistic regression model
log_model <- train(x = X_train, y = y_train, method = "glm", family = binomial, trControl = train_control)

# Train XGBoost model with suppressed warnings
suppressWarnings({
  xgb_model <- train(x = as.matrix(X_train), y = y_train, method = "xgbTree", trControl = train_control, objective = "binary:logistic", silent = TRUE)
})

# Train Random Forest model
rf_model <- train(x = X_train, y = y_train, method = "rf", trControl = train_control)

# Train SVM model
svm_model <- train(x = X_train, y = y_train, method = "svmRadial", trControl = train_control, preProcess = c("center", "scale"))

# Create a list of models to ensemble
models_list <- caretList(
  x = X_train, y = y_train,
  trControl = train_control,
  methodList = c("glm", "rf", "svmRadial", "xgbTree")
)

# Ensemble the models using a greedy algorithm
ensemble_model <- caretEnsemble(models_list, metric = "Accuracy", trControl = train_control)

# Make predictions on the test set using the ensemble model
ensemble_predictions <- predict(ensemble_model, newdata = X_test)




# Calculate accuracy
ensemble_accuracy <- confusionMatrix(ensemble_predictions, y_test)$overall['Accuracy']
print(paste("Ensemble Model Accuracy:", ensemble_accuracy))

# Detailed performance metrics
conf_mat <- confusionMatrix(ensemble_predictions, y_test)
print(conf_mat)

# ROC curve
roc_curve <- roc(response = y_test, predictor = ensemble_probabilities)
roc_auc <- auc(roc_curve)
print(paste("ROC AUC:", roc_auc))
plot(roc_curve, main="ROC Curve", col="#1c61b6")
abline(a=0, b=1, col="red", lty=2)  # Adds a reference line
