# Mentorama Model API

Modelos treinados usando Regressão Linear e KNN

### Rotas
- POST
    ```/predict:``` versão final do modelo para previsão
        
        model inputs:
            -  features: array contendo 16 valores de features
        
        model output: 
        prediction: objeto com atributos 'knn' para predição no modelo knn e 'linReg' para predição no modelo de regressão linear

- GET
    ```/model_health/<model_id>```: Metricas do modelo. Recall e Precision disponíveis. 
        
        model_id: id do modelo em produção