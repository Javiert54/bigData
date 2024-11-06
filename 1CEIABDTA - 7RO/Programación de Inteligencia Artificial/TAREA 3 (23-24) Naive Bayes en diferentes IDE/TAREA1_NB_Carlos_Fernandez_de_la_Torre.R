####################################
# Ejemplo Clasificador Naive Bayes #
####################################
#install.packages("e1071")
library(e1071)
install.packages("tm")
library(tm)
# Vamos a utilizar un dataset de con 435 instancias
# de las cuales 61.37931% son votantes democratas y 38.62069% son republicanos   
###########################
# 1. Preparacion de datos #
###########################
# Al poner header= FALSE se crea la primera fila con los nombres de las columnas V1, V2, ...
vote_data <- read.csv("house-votes-84.data",header=F, stringsAsFactors = FALSE) 
header<-c("NAME","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16")
# Modificamos la primera fila con los nuevos nombres de las columnas 
names(vote_data)<-header
# Observamos el formato del dataset 
head(vote_data)
tail(vote_data)
# Eliminamos los ? por datos vacíos
vote_data[vote_data=="?"]<-"" 
str(vote_data)
# hacemos que la columna NAME sea factor (variable categórica,con valores democrat o republican)
vote_data$NAME <- as.factor(vote_data$NAME)
# Veamos cuantas instancias tenemos de cada clase
prop.table(table(vote_data$NAME))
##############################################
# 2. Creacion de datos de entrenamiento/test #
##############################################
vote_raw_train <- vote_data[1:370, ]
vote_raw_test  <- vote_data[371:435, ]
# Observamos que se mantienen las proporciones
prop.table(table(vote_raw_train$NAME))
prop.table(table(vote_raw_test$NAME))
##########################################
# 3. Creacion de features para el modelo #
##########################################
# Vote_raw_train y vote_raw_test son data frames de variables categorical. 
# Ambos son una estructura que pueda ser utilizada como argumento por el clasificador NB
# Entrenamos un clasificador NB.
# El modelo utiliza la presencia "yes" o ausencia "no" de uno de los 16 restantes atributos
# para estimar la probabilidad de que un votante sea demócrata o republicano.
# laplace=1 para los datos desconocidos 
vote_classifier <- naiveBayes(vote_raw_train, vote_raw_train$NAME, laplace = 1) 
# Predecimos la clase más probable
vote_test_pred <- predict(vote_classifier, vote_raw_test, type = "class") #vote_classifier es el modelo entrenado y vote_raw_test los datos para testear el modelo
# vote_test_pred contiene la predición para los 65 casos de test
# Matriz de confusion
table(vote_test_pred, vote_raw_test$NAME)
prop.table(table(vote_test_pred, vote_raw_test$NAME), margin = 2)
# Predicciones donde se ha equivocado el modelo
vote_raw_test[vote_raw_test$NAME != vote_test_pred,] # Comparamos los datos de las dos columnas
# Predecimos con probabilidades 
vote_test_pred <- predict(vote_classifier, vote_raw_test, type = "raw")
pred <- as.data.frame(vote_test_pred)
head(pred)
#install.packages("ROCR")
library(ROCR) 
pred <- prediction(predictions = pred$democrat, labels = vote_raw_test$NAME)
#Curva ROC 
perf <- performance(pred, "tpr", "fpr") 
plot(perf) 
#Curva Precision-Recall 
perf2 <- performance(pred, "prec", "rec") 
plot(perf2) 
#Curva sensivity-specificity 
perf3 <- performance(pred, "sens", "spec") 
plot(perf3)

