# Stylest.IA Inference Service

Este repositório contém o serviço de inferência Stylest.IA, licenciado sob **AGPL-3.0**, que expõe um endpoint REST para classificação de imagens usando modelos YOLO treinados.

---

## 📋 Sumário

- [Descrição](#descrição)  
- [Pré-requisitos](#pré-requisitos)  
- [Instalação](#instalação)  
- [Estrutura de diretórios](#estrutura-de-diretórios)  
- [Uso](#uso)  
  - [Variáveis de ambiente](#variáveis-de-ambiente)  
  - [Executando o serviço](#executando-o-serviço)  
  - [Endpoint `/classify`](#endpoint-classify)  
- [Contribuição](#contribuição)  
- [Licença](#licença)  

---

## 📖 Descrição

O **Stylest.IA Inference Service** é um micro-serviço FastAPI que carrega modelos YOLO pré-treinados para:

1. **classificationCategory**: classifica a categoria genérica de uma peça de roupa.  
2. **classificationClothes**: classifica o tipo específico de peça de roupa.

O código que importa e chama `ultralytics.YOLO` é licenciado sob AGPL-3.0 e está disponível neste repositório.

---

## ⚙️ Pré-requisitos

- Python 3.9+  
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (instalado via pip)  
- Modelos pré-treinados (`.pt`)  
- pipenv ou venv para gerenciar ambientes  

---

## 🛠️ Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/SeuUsuario/stylest-ia-inference.git
   cd stylest-ia-inference
