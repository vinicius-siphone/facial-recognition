from docx import Document
from transformers import pipeline
import os

# Exemplo simples de sumarização
# from transformer import pipeline
# summarizer = pipeline("summarization")
# text = "Seu texto longo aqui"
# summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
# print(summary)

# Exemplo simples de análise de sentimentos
# from transformer import pipeline
# sentiment_analyzer = pipeline("sentiment-analysis")
# result = classifier("Eu amo este produto!")
# print(result)

# Exemplo simples de tradução automática
# from transformer import pipeline
# translator = pipeline("translation_en_to_fr")
# translation = translator("Hello, how are you?", max_length=40)
# print(translation)

# Carregar o modelo de sumarização
summarizer = pipeline("summarization")

# Função para ler o conteúdo de um arquivo .docx
def read_docx(docx_path):
  document = Document(docx_path)
  full_text = []
  for para in document.paragraphs:
    full_text.append(para.text)
  return '\n'.join(full_text)
  
# Função para resumir o texto usando o modelo de sumarização
def summarize_text(text, max_length=130, min_length=30, do_sample=False):
  summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
  return summary[0]['summary_text']

# Função para salvar o resumo em um arquivo .txt
def save_summary_to_txt(summary_text, txt_path):
  with open(txt_path, 'w', encoding='utf-8') as file:
    file.write(summary_text)

# Função principal para executar o processo de leitura, sumarização e salvamento
if __name__ == "__main__":
  docx_path = 'document.docx'
  txt_path = 'resumo.txt'
  full_text = read_docx(docx_path)
  summary = summarize_text(full_text, max_length=200, min_length=50)
  save_summary_to_txt(summary, txt_path)
  print("Sumarização completa. O resumo foi salvo em 'resumo.txt'")