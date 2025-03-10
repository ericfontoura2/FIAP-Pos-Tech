import pandas as pd
import numpy as np
import faiss
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 0. instalar bibliotecas
# pip install transformers datasets pandas torch faiss-cpu
# pip install 'accelerate>=0.26.0'

# 1. Carregar e limpar o dataset JSON
dataset_path = "/Users/ericfontoura/FIAP/TC 3/trn_50k.json"  # Caminho Arquivo JSON
df = pd.read_json(dataset_path, lines=True)

# Remover linhas onde a coluna "content" está vazia
df = df[df["content"].str.strip() != ""]

# Manter apenas as colunas "uid", "title" e "content"
df = df[["uid", "title", "content"]]

# Adicionar rótulos fictícios (exemplo: 0 para "não relevante", 1 para "relevante")
# Substitua isso por rótulos reais se disponíveis
df["labels"] = np.random.randint(0, 2, size=len(df))  # Rótulos aleatórios para exemplo

# 2. Tokenização dos dados
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Função para tokenizar os títulos e retornar input_ids e attention_mask
def tokenize_titles(title):
    tokenized = tokenizer(title, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    return {
        "input_ids": tokenized["input_ids"].squeeze().numpy(),  # Converte para numpy array
        "attention_mask": tokenized["attention_mask"].squeeze().numpy(),  # Converte para numpy array
    }

# Aplicar a tokenização ao dataset
tokenized_data = df["title"].apply(tokenize_titles)

# Adicionar input_ids e attention_mask ao DataFrame
df["input_ids"] = tokenized_data.apply(lambda x: x["input_ids"])
df["attention_mask"] = tokenized_data.apply(lambda x: x["attention_mask"])

# 3. Converter o DataFrame para o formato de Dataset da Hugging Face
dataset = Dataset.from_pandas(df)

# Dividir o dataset em treino e validação
dataset = dataset.train_test_split(test_size=0.2)

# 4. Fine-Tuning do Modelo BERT
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # 2 classes

# Configurar os argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Substituído evaluation_strategy por eval_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Função para preparar os dados para o Trainer
def preprocess_function(examples):
    return {
        "input_ids": examples["input_ids"],
        "attention_mask": examples["attention_mask"],
        "labels": examples["labels"],  # Adicionar rótulos
    }

# Aplicar a função de pré-processamento ao dataset
dataset = dataset.map(preprocess_function, batched=True)

# Criar o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Treinar o modelo
trainer.train()

# 5. Configuração da Integração RAG
dimension = 768  # Dimensão dos embeddings do BERT
index = faiss.IndexFlatL2(dimension)

# Gerar embeddings dos títulos usando o modelo BERT fine-tuned
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Adicionar embeddings ao índice FAISS
embeddings = np.array([generate_embeddings(title) for title in df["title"]])
index.add(embeddings)

# Função para recuperar títulos relevantes
def retrieve_relevant_titles(query, k=5):
    query_embedding = generate_embeddings(query)
    distances, indices = index.search(query_embedding, k)
    return df.iloc[indices[0]]["title"].tolist()

# 6. Geração de Respostas
def generate_response(query):
    # Recuperar títulos relevantes
    relevant_titles = retrieve_relevant_titles(query)
    
    # Combinar a pergunta com os títulos relevantes
    context = " ".join(relevant_titles)
    prompt = f"Pergunta: {query}\nContexto: {context}\nResposta:"
    
    # Gerar a resposta usando o modelo fine-tuned
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response, relevant_titles

# Testar a geração de respostas
query = "What is the best Italian Cookbook?"
response, sources = generate_response(query)
print("Resposta:", response)
print("Fontes:", sources)
