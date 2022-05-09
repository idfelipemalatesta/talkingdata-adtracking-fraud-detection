#### Objetivo do Projeto ####
# O objetivo é prever se um usuário fará o download de um aplicativo depois de assistir o anúncio.
# O modelo deverá apresentar uma acurácia >= 75
# Fonte dos dados: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/overview

#### Carregando os Pacotes ####

library(dplyr)
library(funModeling)
library(lubridate)
library(caret)
library(ROSE)
library(rpart)

#### Carregando os dados ####

# Importando os dados
dataset <- read.csv("train_sample.csv", na.strings = c("", " "))
novos_dados <- read.csv("test.csv", na.strings = c("", " "))

#### Engenharia de Atributos e Análise Exploratória ####
glimpse(dataset)
glimpse(novos_dados)

dataset %>% df_status()
novos_dados %>% df_status()

# Padronizando as colunas
dataset$ip = NULL
novos_dados$ip = NULL
dataset$attributed_time = NULL

dataset <- dataset %>% mutate(is_train = TRUE)
novos_dados <- novos_dados %>% mutate(is_train = FALSE)

submission <- novos_dados$click_id
novos_dados$click_id = NULL

# Juntando os datasets
dataset_full <- bind_rows(dataset, novos_dados)
glimpse(dataset_full)
dataset_full %>%  df_status()

table(dataset_full$is_attributed) # Verificando o baleanceamento da variável target

# Analisando quais são os aplicativos com mais downloads?
dataset %>% 
  select_all() %>% 
  filter(is_attributed == 1) %>% 
  group_by(app) %>% 
  summarise(qtde = n()) %>% 
  arrange(desc(qtde)) %>%
  head(8)

# Analisando quais são os anuncios com mais downloads?
dataset %>% 
  select_all() %>% 
  filter(is_attributed == 1) %>% 
  group_by(channel) %>% 
  summarise(qtde = n()) %>% 
  arrange(desc(qtde)) %>% 
  head(8)

# Analisando qual o periodo do dia com mais downloads?

# Alterando o tipo da variável click_time para datetime
# Criando novas variáveis através da variável click_time(ano, mes, dia, hora)
dataset_full <- dataset_full %>% 
  mutate(click_time = ymd_hms(click_time),
         ano_f = year(click_time),
         mes_f = month(click_time),
         dia_f = day(click_time),
         hora_f = hour(click_time))

glimpse(dataset_full)
dataset_full %>%  df_status()

# Removendo as variáveis ano_f e mes_f pois ambas apresentam apenas 1 valor no dataset
dataset_full$ano_f = NULL
dataset_full$mes_f = NULL

# Criando uma nova variável que contém os periodos do dia
dataset_full <- dataset_full %>% 
  mutate(periodo_dia = case_when(hora_f <= 5 ~ 'Madrugada',
                                 hora_f <= 11 ~ 'Manha',
                                 hora_f <= 18 ~ 'Tarde',
                                 hora_f <= 23 ~ 'Noite'))

# Periodo do dia com mais downloads
dataset_full %>% 
  select_all() %>% 
  filter(is_attributed == 1) %>% 
  group_by(periodo_dia) %>% 
  summarise(qtde = n()) %>% 
  arrange(desc(qtde))

#### Pré-Processamento ####

# Criando novas variáveis do tipo fator para as variáveis categóricas
dataset_full <- dataset_full %>% 
  mutate(app_f = case_when(app <= 3 ~ 1,
                           app <= 12 ~ 2,
                           TRUE ~ 3),
         device_f = case_when(device == 0 ~ 1,
                              device == 1 ~ 2,
                              TRUE ~ 1),
         os_f = case_when(os <= 16 ~ 1,
                          TRUE ~ 2),
         channel_f = case_when(channel <= 200 ~ 1,
                               channel <= 300 ~ 2,
                               TRUE ~ 3))

glimpse(dataset_full)
dataset_full %>% df_status()

# Alterando o tipo das novas variáveis e da variável target
dataset_full <- dataset_full %>%
  select(app, device, os, channel, click_time, dia_f, hora_f, periodo_dia,
         app_f, device_f, os_f, channel_f, is_train, is_attributed) %>% 
  mutate(app_f = as.factor(app_f),
         device_f = as.factor(device_f),
         os_f = as.factor(os_f),
         channel_f = as.factor(channel_f),
         is_attributed = as.factor(is_attributed))

# Criando uma nova variável númerica para os periodos do dia
dataset_full <- dataset_full %>%
  mutate(cod_periodo_dia = case_when(periodo_dia == 'Madrugada' ~ 1,
                                     periodo_dia == 'Manha' ~ 2,
                                     periodo_dia == 'Tarde' ~ 3,
                                     periodo_dia == 'Noite' ~ 4))

# Organizando a ordem das variáveis e alterando o tipo da variável cod_periodo_dia
dataset_full <- dataset_full %>%
  select(app, device, os, channel, click_time, dia_f, hora_f, periodo_dia, cod_periodo_dia,
         app_f, device_f, os_f, channel_f, is_train, is_attributed) %>% 
  mutate(cod_periodo_dia = as.factor(cod_periodo_dia))


#### Seleção de Variáveis ####
dataset_prep <- dataset_full %>%
  select(app, device, os, channel, click_time, dia_f, hora_f, -periodo_dia, cod_periodo_dia,
         app_f, device_f, os_f, channel_f, is_train, is_attributed)

glimpse(dataset_prep)
dataset_prep %>% df_status()

#### Divisão dos dados ####

# Dividindo o conjunto de dados de treino, teste e os novos dados.
df_treino_teste <- dataset_prep %>% 
  filter(is_train == TRUE) %>% 
  glimpse()

df_treino_teste %>% df_status()

df_novos_dados <- dataset_prep %>% 
  filter(is_train == FALSE) %>% 
  mutate(is_attributed = NULL) %>% 
  glimpse()

df_novos_dados %>% df_status()

# A coluna dia_f nos novos dados apresenta apenas um valor unico.
# Removendo a coluna dia_f dos datasets
df_treino_teste$dia_f = NULL
df_novos_dados$dia_f = NULL

# Removendo a coluna is_train
df_treino_teste$is_train = NULL
df_novos_dados$is_train = NULL

glimpse(df_treino_teste)
df_treino_teste %>% df_status()

glimpse(df_novos_dados)
df_novos_dados %>% df_status()

# Balenceando a variável target
prop.table(table(df_treino_teste$is_attributed))
df_treino_teste <- ovun.sample(is_attributed ~ ., method = "over", data = df_treino_teste)$data
prop.table(table(df_treino_teste$is_attributed))

# Dividindo o conjunto de dados de Treino e Teste
?createDataPartition
indice <- df_treino_teste$is_attributed %>% 
  createDataPartition(p = .75, list = FALSE)

df_treino <- df_treino_teste[indice, ]
df_teste <- df_treino_teste[-indice, ]
glimpse(df_treino)
glimpse(df_teste)
df_treino %>% df_status()
df_teste %>% df_status()

#### Criação e Treino do Modelo ####
# Treinando o modelo de Arvore de Decisão
modelo <- rpart(is_attributed ~ ., 
                data = df_treino, 
                method = "class", 
                parms = list(split = "information"), 
                control = rpart.control(minsplit = 5))

modelo

# Previsões nos dados de treino
pred_treino = predict(modelo, df_treino, type='class')

# Percentual de previsões corretas com dataset de treino
mean(pred_treino==df_treino$is_attributed)

# Previsões nos dados de teste
pred_teste = predict(modelo, df_teste, type='class')

# Percentual de previsões corretas com dataset de teste
mean(pred_teste==df_teste$is_attributed)

# Confusion Matrix
table(pred_teste, df_teste$is_attributed)

# Previsões nos novos dados
pred_novos_dados = predict(modelo, df_novos_dados, type='class')

#### Submission Kaggle e Score do Modelo ####
glimpse(submission)

submission <- cbind(submission, as.integer(pred_novos_dados))
colnames(submission) <- c("click_id", "is_attributed")
class(submission)
submission <- as.data.frame(submission)

submission %>% df_status()
summary(submission$is_attributed)

submission <- submission %>% 
  mutate(is_attributed = if_else(is_attributed == 1, 0, 1))

head(submission)

# Salvando o resultado das predições do modelo nos novos dados
write.csv(submission, file = "submissionTree.csv", row.names = FALSE)

# Score do modelo no Kaggle
# Score: 0.88843

# Salvando o modelo