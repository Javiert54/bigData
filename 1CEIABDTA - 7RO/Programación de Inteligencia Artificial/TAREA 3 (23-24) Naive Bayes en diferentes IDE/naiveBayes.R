library(e1071)
data(iris)  # Usaremos el conjunto de datos iris como ejemplo
set.seed(123)  # Para reproducibilidad
sample_index <- sample(1:nrow(iris), 0.7 * nrow(iris))
train_data <- iris[sample_index, ]
test_data <- iris[-sample_index, ]
model <- naiveBayes(Species ~ ., data = train_data)
predictions <- predict(model, test_data)
confusion_matrix <- table(predictions, test_data$Species)
print(confusion_matrix)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", accuracy))

