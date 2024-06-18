{"metadata":{"kernelspec":{"name":"ir","display_name":"R","language":"R"},"language_info":{"name":"R","codemirror_mode":"r","pygments_lexer":"r","mimetype":"text/x-r-source","file_extension":".r","version":"4.0.5"},"kaggle":{"accelerator":"none","dataSources":[{"sourceId":1731978,"sourceType":"datasetVersion","datasetId":1027924}],"dockerImageVersionId":30618,"isInternetEnabled":true,"language":"r","sourceType":"notebook","isGpuEnabled":false}},"nbformat_minor":4,"nbformat":4,"cells":[{"source":"<a href=\"https://www.kaggle.com/code/pasqualepellegrini/pancreatic-cancer-prediction-models?scriptVersionId=183703900\" target=\"_blank\"><img align=\"left\" alt=\"Kaggle\" title=\"Open in Kaggle\" src=\"https://kaggle.com/static/images/open-in-kaggle.svg\"></a>","metadata":{},"cell_type":"markdown"},{"cell_type":"markdown","source":"<p style=\"padding:15px; background-color:darkgreen; font-family:Verdana; font-weight:bold; color:white; font-size:110%; letter-spacing: 2px; text-align:left; border-radius: 10px 10px\">1. Introduction</p>","metadata":{}},{"cell_type":"markdown","source":"**BACKGROUND**\n\nWith a 5-year survival rate of only 12%, pancreatic cancer is one of the deadliest diseases worldwide. \n\nEarly diagnosis is vital for patient survival given the absence of effective therapies and diagnostic tests.\n\nThe prospect of detecting pancreatic cancer through non-invasive diagnostic approaches, such as blood or urine tests to measure cancer-specific biomarkers, is often regarded as the \"holy grail\" of cancer diagnostics. However, this area of research faces significant challenges due to the complexity of cancer types, patient heterogeneity and lack of biomarker specificity.\n\nAchieving high accuracy in diagnostic tests remains a major obstacle in this endeavor.\n\n**OBJECTIVE**\n\nIn this notebook I have applied some classical maschine learning models to a clinical dataset to predict pancreatic cancer based on clinical data.\n\n**THE DATA**\n\nThe dataset is very well balanced as it includes 103 healthy controls, 208 non-cancer conditions and 199 cancer cases. It includes features such as patient demographics (sex and age) and levels of protein biomarkers (creatinine, LYVE1, REG1B, and TFF1). It is important to note that features directly related to cancer, such as the stage of progression, are excluded from the prediction model.","metadata":{}},{"cell_type":"markdown","source":"<p style=\"padding:15px; background-color:darkgreen; font-family:Verdana; font-weight:bold; color:white; font-size:110%; letter-spacing: 2px; text-align:left; border-radius: 10px 10px\">2. Exploratory data analysis</p>","metadata":{}},{"cell_type":"code","source":"# Libraries for data manipulation/analysis\nlibrary(tidyverse)\nlibrary(data.table)\nlibrary(patchwork)\nlibrary(ggcorrplot)\nlibrary(glue)\nlibrary(nnet)\n\n# ML libraries\nlibrary(randomForest)\nlibrary(xgboost)\nlibrary(caret)\nlibrary(caretEnsemble)\nlibrary(e1071)\nlibrary(pROC)","metadata":{"execution":{"iopub.status.busy":"2024-06-14T15:18:06.206974Z","iopub.execute_input":"2024-06-14T15:18:06.209143Z","iopub.status.idle":"2024-06-14T15:18:06.249047Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"# Read the dataset\ndt = fread(\"/kaggle/input/urinary-biomarkers-for-pancreatic-cancer/Debernardi et al 2020 data.csv\")\nhead(dt, 10)","metadata":{"execution":{"iopub.status.busy":"2024-06-14T14:22:45.319756Z","iopub.execute_input":"2024-06-14T14:22:45.355139Z","iopub.status.idle":"2024-06-14T14:22:45.446172Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"######\n# QC #\n######\n\n# Replace missing values with NA for numeric columns\ndt[, (names(dt)) := lapply(.SD, function(x) ifelse(is.na(x) | x == \"\", NA, x)), .SDcols = names(dt)]\n\n# Check if missing values are replaced\ncolSums(is.na(dt))\n","metadata":{"execution":{"iopub.status.busy":"2024-06-14T14:22:48.625012Z","iopub.execute_input":"2024-06-14T14:22:48.627055Z","iopub.status.idle":"2024-06-14T14:22:48.663057Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"**NOTES**: Since 40% of the CA19.9 values are missing, (240 out of 590) we could consider dropping this column. However CA19.9 is one of the most used biomarkers for pancreatic cancer so it might be worth keeping it and replace missing values. Removing the rows is not an option given the limited size of the observations (590).","metadata":{}},{"cell_type":"code","source":"# Data types and structure\nstr(dt)","metadata":{"execution":{"iopub.status.busy":"2024-06-14T12:54:28.878484Z","iopub.execute_input":"2024-06-14T12:54:28.880187Z","iopub.status.idle":"2024-06-14T12:54:28.936693Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"# Convert diagnosis column to categorical\ndt$diagnosis <- as.character(dt$diagnosis)","metadata":{"execution":{"iopub.status.busy":"2024-06-14T12:54:28.941079Z","iopub.execute_input":"2024-06-14T12:54:28.942764Z","iopub.status.idle":"2024-06-14T12:54:28.958376Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"#Shapiro-Wilk normality test\nfor (name in names(dt)) {\n    if (is.numeric(dt[[name]])) {\n        result <- shapiro.test(dt[[name]])\n        if (result$p.value < 0.05) {\n            print(paste(name, \"is NOT normally distributed.\", \" p.value:\",\n                        format(result$p.value, digits = 4)))\n        } else {\n            print(paste(name, \"is normally distributed.\",\" p.value:\",\n                        format(result$p.value, digits = 4)))\n        }\n    }\n}\n","metadata":{"execution":{"iopub.status.busy":"2024-06-14T12:54:28.962737Z","iopub.execute_input":"2024-06-14T12:54:28.964547Z","iopub.status.idle":"2024-06-14T12:54:29.003282Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"**NOTES**: None of the numeric columns show normal distribution","metadata":{}},{"cell_type":"code","source":"#############################################\n# Display distribution of numerical columns #\n#############################################\n\n# Filter specific columns\nvariables_to_keep <- c(\"age\", \"plasma_CA19_9\", \"LYVE1\", \"REG1A\", \"TFF1\", \"creatinine\", \"REG1B\")\ndt_filtered <- dt[,..variables_to_keep, with = FALSE]\n\n# Reshape data into long format for ggplot\ndt_long <- pivot_longer(dt_filtered, cols = everything(), names_to = \"variable\", values_to = \"value\")\n\n# Convert the \"value\" column to numeric if it's not already numeric\ndt_long$value <- as.numeric(dt_long$value)\n\n# Make plots for each variable\n# Used lapply to make a list of plots\n\nplots_numeric <- lapply(unique(dt_long$variable), function(var) {\n    ggplot(subset(dt_long, variable == var),\n          aes(x = value)) +\n    geom_histogram(binwidth = 1, fill = \"skyblue\", color = \"black\") +\n    labs(title = paste(var),\n        x = var, y = \"Frequency\") +\n    theme_minimal()\n})\n\n# Combine plots in one panel\ncombined_plots <- wrap_plots(plots_numeric, ncol = 3)\nprint(combined_plots)","metadata":{"execution":{"iopub.status.busy":"2024-06-14T14:23:33.177412Z","iopub.execute_input":"2024-06-14T14:23:33.179943Z","iopub.status.idle":"2024-06-14T14:23:48.539876Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"**NOTES**: The histograms above are just to visualize how the data is distributed, they clearly do not follow a normal distribution. The age might be normally distributed but probably the n is not high enough to get a significant value in the Shapiro test.\n","metadata":{}},{"cell_type":"code","source":"# Plot the balance of categorical columns\ncategorical_columns <- c('sex','diagnosis','sample_origin','patient_cohort')\n\nplots_categorical <- lapply(categorical_columns, function(var) {\n    \n    # count the occurrencies of each category\n    category_count <- table(dt[[var]])\n\n    # convert counts to a data frame for plotting\n    category_data <- data.frame(\n        category = names(category_count),\n        count = as.numeric(category_count))\n    \n    # bar plot\n    ggplot(category_data, aes(x = category, y = count)) +\n    geom_bar(stat = \"identity\", fill = \"skyblue\", color = \"black\") +\n    labs(title = paste(\"Frequency of\",var), x = \"Category\", y = \"N\") +\n    theme_minimal()\n    \n})\n\n# combine plots in one panel\ncombined_plots <- wrap_plots(plots_categorical, ncol = 2)\nprint(combined_plots)","metadata":{"execution":{"iopub.status.busy":"2024-06-14T14:23:54.40754Z","iopub.execute_input":"2024-06-14T14:23:54.409435Z","iopub.status.idle":"2024-06-14T14:23:55.192816Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"**NOTES**: Sex, diagnosis and patient cohort are relatively well balanced.","metadata":{}},{"cell_type":"code","source":"########################\n# CORRELATION ANALYSIS #\n########################\n\n# Define the categorical and numerical columns to include\ncategorical_columns <- c('sex', 'diagnosis', 'sample_origin', 'patient_cohort')\nnumeric_column_names <- c(\"age\", \"plasma_CA19_9\", \"creatinine\", \"LYVE1\", \"REG1B\", \"TFF1\", \"REG1A\")\n\n# Subset the data table to get categorical columns\ncategorical_data_df <- dt[, .SD, .SDcols = categorical_columns]\n\n# Convert all columns to factors to ensure that the categorical data is ready for one-hot encoding\ncategorical_data_df[] <- lapply(categorical_data_df, as.factor)\n\n# Convert categorical data to model matrix (one-hot encoding)\ndt_one_hot <- model.matrix(~ . - 1, data = categorical_data_df)  # '- 1' removes the intercept\n\n# Convert the model matrix to a data frame\ndt_one_hot_df <- as.data.frame(dt_one_hot)\n\n# Combine numerical columns with the one-hot encoded categorical columns\nnumeric_data_df <- dt[, .SD, .SDcols = numeric_column_names]\ndt_combined <- cbind(numeric_data_df, dt_one_hot_df)\n\n# Impute missing values with the column median for numerical columns (it's more robust for non-normal distribution)\ndt_combined <- dt_combined %>% \n  mutate(across(all_of(numeric_column_names), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))\n\n# Remove columns with zero variance\ndt_combined <- dt_combined %>% select(where(~ var(.) > 0))\n\n# Compute Pearson correlation coefficients\ncorrelation_matrix <- cor(dt_combined, use = \"pairwise.complete.obs\", method = \"pearson\")\n\n# Plot the correlation matrix\nggcorrplot(correlation_matrix, method = \"circle\", type = \"upper\", \n           lab = TRUE, lab_size = 3, colors = c(\"blue\", \"white\", \"red\"), \n           title = \"Correlation Matrix\", hc.order = TRUE)\n","metadata":{"execution":{"iopub.status.busy":"2024-06-14T14:23:59.071856Z","iopub.execute_input":"2024-06-14T14:23:59.073918Z","iopub.status.idle":"2024-06-14T14:23:59.64361Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"**Notes**: Overall there is no strong correlation between variables. ","metadata":{}},{"cell_type":"markdown","source":"<p style=\"padding:15px; background-color:darkgreen; font-family:Verdana; font-weight:bold; color:white; font-size:110%; letter-spacing: 2px; text-align:left; border-radius: 10px 10px\">3. Models for multiclass classification</p>","metadata":{}},{"cell_type":"code","source":"#######################\n# Logistic Regression #\n#######################\n\n# Prepare the data:\n# Assuming diagnosis1 is the case where both diagnosis2 and diagnosis3 are 0\ndt_combined$diagnosis1 <- as.integer(dt_combined$diagnosis2 == 0 & dt_combined$diagnosis3 == 0)\n\n# Define X and y\nX <- select(dt_combined, -starts_with(\"diagnosis\"))\ny <- apply(dt_combined[, c(\"diagnosis1\", \"diagnosis2\", \"diagnosis3\")], 1, which.max) - 1  # adjust index to start from 0\n\n# Split the data into training and testing sets\nset.seed(42)\ntrain_index <- createDataPartition(y, p = 0.7, list = FALSE)\nX_train <- X[train_index, ]\ny_train <- y[train_index]\nX_test <- X[-train_index, ]\ny_test <- y[-train_index]\n\n# Fit logistic regression model\nmodel <- multinom(as.factor(y_train) ~ ., data = X_train)\n\n# Make predictions\npredictions <- predict(model, newdata = X_test)\n\n# Ensure predictions and actual test labels are factors with the same levels\nall_levels <- sort(unique(c(y_train, y_test)))  # Get all levels from both training and testing sets\n\npredictions_factor <- factor(predictions, levels = all_levels)\ny_test_factor <- factor(y_test, levels = all_levels)\n\n# Evaluate the model\naccuracy <- mean(predictions_factor == y_test_factor)\nprint(glue(\"Accuracy: {accuracy * 100}%\"))\n\n# Detailed performance metrics using confusion matrix\nconf_mat <- confusionMatrix(predictions_factor, y_test_factor)\nprint(conf_mat)\n\n# Additional performance metrics\nprint(conf_mat$byClass) ","metadata":{"execution":{"iopub.status.busy":"2024-06-14T14:24:03.815908Z","iopub.execute_input":"2024-06-14T14:24:03.81818Z","iopub.status.idle":"2024-06-14T14:24:03.960909Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"**Notes**: Logistic regression shows a relatively good accuracy (76%). Cancer cases shows relatively high sensitivity (84.75%) and specificity (91.45%).","metadata":{}},{"cell_type":"code","source":"###########\n# Xgboost #\n###########\n\n# Define X and y\nX <- select(dt_combined, -starts_with(\"diagnosis\"))\ny <- apply(dt_combined[, c(\"diagnosis1\", \"diagnosis2\", \"diagnosis3\")], 1, which.max) - 1\n\n# Split the data into training and testing sets\nset.seed(42)\ntrainIndex <- createDataPartition(y, p = 0.67, list = FALSE)\nX_train <- X[trainIndex, ]\nX_test <- X[-trainIndex, ]\ny_train <- y[trainIndex]\ny_test <- y[-trainIndex]\n\n# Ensure y_train and y_test are numeric\ny_train <- as.numeric(y_train)\ny_test <- as.numeric(y_test)\n\n# library(warnings)\noptions(warn = 1)\n\n# Convert the data to DMatrix format\ndtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)\ndtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)\n\n# Hyperparameter tuning\ntune_grid <- expand.grid(\n  nrounds = c(50, 100, 150),\n  eta = c(0.01, 0.1),\n  max_depth = c(3, 4),\n  gamma = c(0, 1),\n  colsample_bytree = c(0.5, 0.7, 1.0),\n  min_child_weight = c(1, 3, 5),\n  subsample = c(0.5, 0.7, 1.0)\n)\n\n# Train control with cross-validation\ntrain_control <- trainControl(method = \"cv\", number = 5, verboseIter = FALSE)\n\n# Train the XGBoost model with tuning\nxgb_tune <- train(\n  x = as.matrix(X_train),\n  y = y_train,\n  method = \"xgbTree\",\n  trControl = train_control,\n  tuneGrid = tune_grid,\n  objective = \"multi:softmax\",\n  num_class = length(unique(y)), \n  iteration_range = c(1, max(tune_grid$nrounds)),\n  verbosity = 0,\n  verbose = FALSE\n)\n           \n# Best model\nbest_model <- xgb_tune$finalModel\n\n# Make predictions\npredictions <- predict(best_model, newdata = as.matrix(X_test), iteration_range = c(1, best_model$niter))\npredictions <- as.numeric(predictions)\n\n# Convert predictions and y_test to factors with the same levels\npredictions <- factor(predictions, levels = 0:(length(unique(y)) - 1))\ny_test_factor <- factor(y_test, levels = 0:(length(unique(y)) - 1))\n\n# Evaluate the model\nconfusion_mtx <- confusionMatrix(predictions, y_test_factor)\n\n","metadata":{"execution":{"iopub.status.busy":"2024-06-14T14:24:21.560602Z","iopub.execute_input":"2024-06-14T14:24:21.562458Z","iopub.status.idle":"2024-06-14T14:28:06.46344Z"},"collapsed":true,"jupyter":{"outputs_hidden":true},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"# Extract accuracy and other parameters from the confusion matrix\naccuracy <- confusion_mtx$overall[\"Accuracy\"]\nprint(glue(\"Accuracy: {accuracy * 100}%\"))\n\n# Print class-wise metrics\nprint(confusion_mtx$byClass)","metadata":{"execution":{"iopub.status.busy":"2024-06-14T14:28:54.362121Z","iopub.execute_input":"2024-06-14T14:28:54.364176Z","iopub.status.idle":"2024-06-14T14:28:54.391384Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"**NOTES**: Xgboost accuracy is 74%.","metadata":{}},{"cell_type":"code","source":"# Extract feature importance\nimportance_matrix <- xgb.importance(feature_names = colnames(X_train), model = best_model)\n\n# Plot the feature importance\nggplot(importance_matrix, aes(x = reorder(Feature, -Gain), y = Gain)) +\n  geom_bar(stat = \"identity\", fill = \"skyblue\") +\n  labs(x = \"Features\", y = \"Importance\", title = \"Feature Importance\") +\n  theme(axis.text.x = element_text(angle = 90, hjust = 1))","metadata":{"execution":{"iopub.status.busy":"2024-06-14T14:28:59.017374Z","iopub.execute_input":"2024-06-14T14:28:59.020175Z","iopub.status.idle":"2024-06-14T14:28:59.409816Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"**NOTE**: This plot confirms the importance of CA19.9 for pancreatic cancer predicition even though it is known for lacking in specificity.","metadata":{}},{"cell_type":"code","source":"###########################\n# Support Vector Machines #\n###########################\n\n# Define training control\ntrain_control <- trainControl(\n  method = \"cv\",\n  number = 10,\n  savePredictions = \"final\",\n  classProbs = FALSE,\n  verboseIter = FALSE,\n  summaryFunction = defaultSummary\n)\n\n# Define the parameter grid for tuning\ntune_grid <- expand.grid(\n  C = 2^(-5:2),  # Cost parameter\n  sigma = 2^(-5:2)  # RBF kernel parameter\n)\n\n# Ensure y_train and y_test are factors\ny_train <- factor(y_train)\ny_test <- factor(y_test)\n\n# Train the SVM model with RBF kernel\nsvm_model <- train(\n  x = X_train,\n  y = y_train,\n  method = \"svmRadial\",\n  trControl = train_control,\n  tuneGrid = tune_grid,\n  preProcess = c(\"center\", \"scale\")\n)\n\n# Print the best model's tuning parameters\ncat(\"Best Model Parameters:\\n\")\nprint(svm_model$bestTune)\n\n# Make predictions using the correct type\npredictions <- predict(svm_model, newdata = X_test, type = \"raw\")\n\n# Convert predictions to a factor with the same levels as y_train for proper evaluation\npredictions_factor <- factor(predictions, levels = levels(factor(y_train)))\ny_test_factor <- factor(y_test, levels = levels(factor(y_train)))\n\n# Evaluate the model using the confusion matrix\nconf_mat <- confusionMatrix(predictions_factor, y_test_factor)\nprint(conf_mat)\n","metadata":{"execution":{"iopub.status.busy":"2024-06-14T14:29:13.77887Z","iopub.execute_input":"2024-06-14T14:29:13.780931Z","iopub.status.idle":"2024-06-14T14:30:26.401808Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"**Notes**: SVM: 73% accuracy","metadata":{}},{"cell_type":"markdown","source":"<p style=\"padding:15px; background-color:darkgreen; font-family:Verdana; font-weight:bold; color:white; font-size:110%; letter-spacing: 2px; text-align:left; border-radius: 10px 10px\">3. Models for binary classification</p>","metadata":{}},{"cell_type":"markdown","source":"Since the multiclass models did not perform high enough for a diagnostic test I have tried to simplify the classification by grouping the healthy and non-cancer pathologies into one group to have a binary classification of 0 (no cancer) and 1 (cancer).","metadata":{}},{"cell_type":"code","source":"# Remove diagnosis1 and diagnosis2 columns and use only diagnosis3 as target\ndt_combined_binary <- dt_combined %>% select(-diagnosis1, -diagnosis2)","metadata":{"execution":{"iopub.status.busy":"2024-06-14T14:42:08.591438Z","iopub.execute_input":"2024-06-14T14:42:08.593888Z","iopub.status.idle":"2024-06-14T14:42:08.622622Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"#######################\n# Logistic regression #\n#######################\n\n# Ensure target is a factor for binary classification\ndt_combined_binary$diagnosis3 <- as.factor(dt_combined_binary$diagnosis3)\n\n# Define X and y\nX <- select(dt_combined_binary, -diagnosis3)\ny <- dt_combined_binary$diagnosis3\n\n# Train-Test split\nset.seed(42)\ntrain_index <- createDataPartition(y, p = 0.7, list = FALSE)\nX_train <- X[train_index, ]\ny_train <- y[train_index]\nX_test <- X[-train_index, ]\ny_test <- y[-train_index]\n\n# Fit logistic regression model\nmodel <- glm(y_train ~ ., data = X_train, family = binomial)\n\n# Make predictions\npredictions <- predict(model, newdata = X_test, type = \"response\")\npredicted_classes <- ifelse(predictions > 0.5, 1, 0)\n\n# Convert predictions and y_test to factors\npredicted_classes <- as.factor(predicted_classes)\ny_test_factor <- as.factor(y_test)\n\n# Evaluate the model\naccuracy <- mean(predicted_classes == y_test_factor)\nprint(glue(\"Accuracy: {accuracy * 100}%\"))\n\n# Detailed performance metrics\nconf_mat <- confusionMatrix(predicted_classes, y_test_factor)\nprint(conf_mat)\n\n","metadata":{"execution":{"iopub.status.busy":"2024-06-14T14:42:10.616572Z","iopub.execute_input":"2024-06-14T14:42:10.618591Z","iopub.status.idle":"2024-06-14T14:42:10.727994Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"**Notes**: 90% accuracy for binary logistic regression.","metadata":{}},{"cell_type":"code","source":"##############################################\n# Ensemble method for multiple binary models #\n##############################################\n\n# Suppress xgboost warnings\noptions(warn = -1)\n\n# Ensure diagnosis3 is a factor for binary classification and rename levels\ndt_combined_binary$diagnosis3 <- as.factor(dt_combined_binary$diagnosis3)\nlevels(dt_combined_binary$diagnosis3) <- make.names(levels(dt_combined_binary$diagnosis3))\n\n# Define X and y\nX <- select(dt_combined_binary, -diagnosis3)\ny <- dt_combined_binary$diagnosis3\n\n# Split the data into training and testing sets\nset.seed(42)\ntrain_index <- createDataPartition(y, p = 0.7, list = FALSE)\nX_train <- X[train_index, ]\ny_train <- y[train_index]\nX_test <- X[-train_index, ]\ny_test <- y[-train_index]\n\n# Define train control\ntrain_control <- trainControl(method = \"cv\", number = 5, savePredictions = \"final\", classProbs = TRUE, verboseIter = FALSE)\n\n# Train logistic regression model\nlog_model <- train(x = X_train, y = y_train, method = \"glm\", family = binomial, trControl = train_control)\n\n# Train XGBoost model with suppressed warnings\nsuppressWarnings({\n  xgb_model <- train(x = as.matrix(X_train), y = y_train, method = \"xgbTree\", trControl = train_control, objective = \"binary:logistic\", silent = TRUE)\n})\n\n# Train Random Forest model\nrf_model <- train(x = X_train, y = y_train, method = \"rf\", trControl = train_control)\n\n# Train SVM model\nsvm_model <- train(x = X_train, y = y_train, method = \"svmRadial\", trControl = train_control, preProcess = c(\"center\", \"scale\"))\n\n# Create a list of models to ensemble\nmodels_list <- caretList(\n  x = X_train, y = y_train,\n  trControl = train_control,\n  methodList = c(\"glm\", \"rf\", \"svmRadial\", \"xgbTree\")\n)\n\n# Ensemble the models using a greedy algorithm\nensemble_model <- caretEnsemble(models_list, metric = \"Accuracy\", trControl = train_control)\n\n# Make predictions on the test set using the ensemble model\nensemble_predictions <- predict(ensemble_model, newdata = X_test)\n\n\n","metadata":{"execution":{"iopub.status.busy":"2024-06-14T14:54:05.883397Z","iopub.execute_input":"2024-06-14T14:54:05.88532Z","iopub.status.idle":"2024-06-14T14:55:11.399619Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"# Calculate accuracy\nensemble_accuracy <- confusionMatrix(ensemble_predictions, y_test)$overall['Accuracy']\nprint(paste(\"Ensemble Model Accuracy:\", ensemble_accuracy))\n\n# Detailed performance metrics\nconf_mat <- confusionMatrix(ensemble_predictions, y_test)\nprint(conf_mat)\n\n# ROC curve\nroc_curve <- roc(response = y_test, predictor = ensemble_probabilities)\nroc_auc <- auc(roc_curve)\nprint(paste(\"ROC AUC:\", roc_auc))\nplot(roc_curve, main=\"ROC Curve\", col=\"#1c61b6\")\nabline(a=0, b=1, col=\"red\", lty=2)  # Adds a reference line","metadata":{"execution":{"iopub.status.busy":"2024-06-14T15:19:25.88524Z","iopub.execute_input":"2024-06-14T15:19:25.887275Z","iopub.status.idle":"2024-06-14T15:19:26.016288Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"Ensemble accuracy is 88%. Very high ROC: 0.94","metadata":{}},{"cell_type":"markdown","source":"<p style=\"padding:15px; background-color:darkgreen; font-family:Verdana; font-weight:bold; color:white; font-size:110%; letter-spacing: 2px; text-align:left; border-radius: 10px 10px\">4. Final remarks</p>","metadata":{}},{"cell_type":"markdown","source":"**FINAL REMARKS AND CONCLUSIONS**\n\n* Multiclass classification shows modest accuracy (over 70%) across different models, including logistic regression, SVM, and XGBoost. Although the classes were well-balanced, the number of observations was relatively small (around 200 per class). A larger dataset would likely result in improved performance.\n* Simplifying the classification to a binary model increased the accuracy up to 90% and yielded an excellent ROC curve (AUC of 0.94).\n* I analyzed feature importance using XGBoost Gain, which demonstrated that the feature CA19.9 significantly improves model accuracy when used for splits. For this reason, removing the CA19.9 column would likely be detrimental. Since the distribution of CA19.9 is not normal, NAs were replaced with the median. This choice could be further debated in future analyses.\n\n\n**REFERENCES**\n\n* https://www.kaggle.com/datasets/johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer\n\n* https://www.hopkinsmedicine.org/health/conditions-and-diseases/pancreatic-cancer/pancreatic-cancer-prognosis\n\n* O'Neill et al. \"Biomarkers in the diagnosis of pancreatic cancer: Are we closer to finding the golden ticket?\" World J Gastroenterol. 2021 Jul 14;27(26):4045-4087. doi: 10.3748/wjg.v27.i26.4045.\n\n\n**ACKNOWLEDGEMENTS**\n\nI would like to express my gratitude to Bernardi and colleagues for their generous data sharing, which promotes the principles of open science.","metadata":{}}]}