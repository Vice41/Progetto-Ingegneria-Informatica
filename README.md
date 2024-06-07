# Analisi della Fairness di un Modello di Machine Learning

Questo progetto guida l'utente, attraverso un'interfaccia grafica, nell'analisi della fairness di un determinato modello di machine learning.

## Contenuti

- [Relazione del Progetto](https://github.com/Vice41/Progetto-Ingegneria-Informatica/blob/master/Relazione%20Progetto%20Ingegneria%20Informatica.pdf)
- [Training del Modello di Machine Learning](https://github.com/Vice41/Progetto-Ingegneria-Informatica/blob/master/colab%20notebook%20analysis%20of%20diabetes%20in%20women.ipynb)
- [Demo del Chatbot](https://github.com/Vice41/Progetto-Ingegneria-Informatica/blob/master/Chatbot%20demo.mp4)

## Eseguire il Chatbot Localmente

Per eseguire il chatbot localmente, seguire questi passaggi:

1. **Scaricare la repository:**

   ```bash
   git clone https://github.com/Vice41/Progetto-Ingegneria-Informatica.git
   cd Progetto-Ingegneria-Informatica
   ```

2. **Installare i requisiti:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Connettere Azure Open AI:**

   Creare un file `.env` contenente la chiave e l'endpoint al proprio modello:

   ```plaintext
   AZURE_OPENAI_API_KEY=sostituisci_con_la_chiave
   AZURE_OPENAI_ENDPOINT=sostituisci_con_endpoint
   ```

4. **Avviare il sito in locale:**

   ```bash
   flask --app app run
   ```

