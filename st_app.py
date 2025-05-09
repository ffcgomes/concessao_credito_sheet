import os
import numpy as np
import pandas as pd
import joblib
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import streamlit as st
import json

# --- App Configuration ---
st.set_page_config(page_title="Calculadora de Probabilidade de Pagamento Automática")

# --- Hardcoded Google Sheets API Configuration ---
# !!! REPLACE WITH YOUR ACTUAL SPREADSHEET ID !!!
# Example: SPREADSHEET_ID = '1Y0mleR7AdIFsFFgd7Wwt5HhzcdU1DlBTn9LCkYlcAxY'
SPREADSHEET_ID = '1Y0mleR7AdIFsFFgd7Wwt5HhzcdU1DlBTn9LCkYlcAxY' # <<-- COLOQUE O ID DA SUA PLANILHA AQUI

# --- Range de Leitura Inicial (agora inclui até onde a probabilidade será escrita) ---
# Isso é necessário para que a API permita escrever nessa coluna depois.
# Você pode usar A:G para ler todas as linhas até a coluna G, ou A1:G11 para um range fixo.
RANGE_NAME = 'Página1!A:G' # <<-- LER COLUNAS A ATÉ G em TODAS as linhas. Ajuste 'Página1' se necessário.

SCOPES = ['https://www.googleapis.com/auth/spreadsheets'] # Read and write access

# --- File Paths (adjust as needed for your deployment) ---
MODEL_PATH = "./resultados_parciais/modelo_logistico.pkl" # <<-- Verifique se o caminho do modelo está correto

# --- Streamlit Secrets Key ---
# This key must match the section name in your .streamlit/secrets.toml
GCP_SECRETS_KEY = 'gcp_service_account' # <<-- Verifique se a chave nos secrets.toml é essa

# --- Mappings ---
# Mapping of sheet column names to model feature names
COLUMN_MAPPING = {
    'Valor_Parcela': 'ValorQuitacao',
    'Quant_Boletos_Pagos': 'Quant_Pagamentos_Via_Boleto',
    'Idade': 'Quant_Ocorrencia'
}

# Inverse mapping (used internally for header logic, mainly for error messages)
REVERSE_MAPPING = {v: k for k, v in COLUMN_MAPPING.items()}


# --- Helper Functions ---

@st.cache_resource
def load_model_and_encoder(model_path):
    """Loads the pre-trained model and encoder, caching the result."""
    try:
        model, encoder = joblib.load(model_path)
        st.success("Modelo e Encoder carregados com sucesso!")
        return model, encoder
    except FileNotFoundError:
        st.error(f"Erro: Arquivo do modelo ou encoder não encontrado em {model_path}")
        st.stop() # Stop the app execution if essential files are missing
    except Exception as e:
        st.error(f"Erro ao carregar modelo e encoder: {e}")
        st.stop()

def get_google_sheets_service():
    """Authenticates and returns the Google Sheets service object."""
    try:
        # Use credentials from Streamlit secrets
        creds = Credentials.from_service_account_info(
            st.secrets[GCP_SECRETS_KEY], scopes=SCOPES
        )
        service = build('sheets', 'v4', credentials=creds)
        # st.success("Autenticação com Google Sheets bem-sucedida!") # Optional: show success message
        return service
    except KeyError:
        st.error(f"Erro de configuração: Chave '{GCP_SECRETS_KEY}' não encontrada nos Streamlit Secrets.")
        st.info("Por favor, configure seu arquivo .streamlit/secrets.toml com as credenciais do Google Service Account.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao autenticar com Google Sheets: {e}")
        st.stop()


def process_sheet_data(sheet_id, range_name):
    """Reads data, processes it, calculates probabilities, and returns updated data."""

    service = get_google_sheets_service()
    model, encoder = load_model_and_encoder(MODEL_PATH)

    st.info(f"Lendo dados da planilha: {sheet_id} - {range_name}")

    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range=range_name
        ).execute()
        values = result.get('values', [])
    except Exception as e:
        st.error(f"Erro ao ler dados da planilha: {e}")
        return None # Indicate failure

    if not values:
        st.warning('Nenhum dado encontrado na planilha no intervalo especificado.')
        return [] # Retorna lista vazia se nenhum valor lido

    header_original = values[0]
    data_rows = values[1:] # Dados a partir da segunda linha

    if not data_rows:
         st.warning('Planilha contém apenas o cabeçalho ou está vazia após o cabeçalho no intervalo especificado.')
         # Retorna apenas o cabeçalho lido mais a nova coluna 'Probabilidade'
         return [header_original + ['Probabilidade']]


    # --- Mapear colunas da planilha para nomes de features DO MODELO ---
    # Precisamos encontrar os índices no cabeçalho original APENAS para as colunas
    # que o script lê DIRETAMENTE da planilha ANTES da codificação (numéricas + UF).
    # Não procuramos pelos nomes de features one-hot encoded (UF_AL, etc.) no cabeçalho.

    # Definir as features do modelo que *correspondem diretamente* a colunas na planilha
    model_features_from_sheet = [
        'ValorQuitacao',
        'Atraso',
        'Quant_Pagamentos_Via_Boleto',
        'Quant_Ocorrencia',
        'UF' # A coluna 'UF' é lida diretamente, depois codificada
    ]

    # Construir o mapeamento de nomes de features do modelo (vindo da planilha) para índices no cabeçalho original
    idx_map_original = {}
    for model_feature_name in model_features_from_sheet:
         sheet_col_name_to_find = model_feature_name # Assume o nome da planilha é o mesmo por padrão

         # Verificar se esta feature do modelo é um VALOR no nosso COLUMN_MAPPING
         is_mapped = False
         for original_sheet_name, mapped_model_name in COLUMN_MAPPING.items():
              if mapped_model_name == model_feature_name:
                   sheet_col_name_to_find = original_sheet_name # Encontrou o nome da planilha
                   is_mapped = True
                   break

         # Lidar com a coluna 'UF' explicitamente
         if model_feature_name == 'UF':
              sheet_col_name_to_find = 'UF'
              is_mapped = True

         # Tentar encontrar o índice desta coluna no cabeçalho lido
         try:
              idx_map_original[model_feature_name] = header_original.index(sheet_col_name_to_find)
         except ValueError:
              error_msg = f"Erro: Coluna esperada '{sheet_col_name_to_find}' não encontrada no cabeçalho da planilha lido."
              if not is_mapped:
                   error_msg = f"Erro: Coluna esperada '{model_feature_name}' não encontrada no cabeçalho da planilha lido."

              st.error(error_msg)
              st.info(f"Cabeçalho lido da planilha: {header_original}")
              st.info(f"Procurando no cabeçalho por colunas que correspondem a estas features do modelo: {model_features_from_sheet}")
              return None

    # Prepare list for updating (includes header with new column)
    updated_header = header_original[:]
    if 'Probabilidade' not in updated_header:
         updated_header.append('Probabilidade')

    updated_values = [updated_header]


    st.info(f"Processando {len(data_rows)} linhas...")
    progress_bar = st.progress(0)

    processed_rows_count = 0
    errored_rows_count = 0

    # Process each row to calculate probability
    for i, row in enumerate(data_rows):
        row_processed = row[:]
        while len(row_processed) < len(header_original):
            row_processed.append('')

        prob_value = ''
        row_error = False

        # --- Ler e Validar Dados da Linha ---
        input_data_for_encoding = {}
        try:
            for model_feature_name in model_features_from_sheet:
                 original_idx = idx_map_original[model_feature_name]
                 value = str(row_processed[original_idx]).strip()

                 if not value:
                      row_error = True
                      break # Dado essencial vazio

                 input_data_for_encoding[model_feature_name] = value

            if not row_error: # Somente se todos os essenciais estiverem presentes
                try:
                     # Converter numéricos, lidar com vírgula decimal
                     valor_quitacao = float(input_data_for_encoding['ValorQuitacao'].replace(',', '.'))
                     atraso = float(input_data_for_encoding['Atraso'].replace(',', '.'))
                     quant_boletos = float(input_data_for_encoding['Quant_Pagamentos_Via_Boleto'].replace(',', '.'))
                     quant_ocorrencia = float(input_data_for_encoding['Quant_Ocorrencia'].replace(',', '.'))
                     uf = input_data_for_encoding['UF']

                except ValueError as ve:
                     st.warning(f"Erro de conversão de tipo na linha {i+2}: {ve}. Verifique os valores numéricos.")
                     row_error = True

            # Codificar UF
            if not row_error:
                 uf_for_encoding = np.array([[uf]])
                 try:
                     uf_encoded = encoder.transform(uf_for_encoding) # Sem .toarray()
                     if hasattr(encoder, 'handle_unknown') and encoder.handle_unknown == 'error':
                          ufs_categories = encoder.categories_[0].tolist()
                          if uf not in ufs_categories:
                               raise ValueError(f"UF '{uf}' não vista durante o treinamento do encoder.")
                 except Exception as e:
                     st.warning(f"Erro durante a codificação da UF '{uf}' na linha {i+2}: {e}")
                     row_error = True

            # --- Montar Vetor FINAL para o Modelo ---
            if not row_error:
                input_vector_list = []
                processed_numerical_values = {
                    'ValorQuitacao': valor_quitacao,
                    'Atraso': atraso,
                    'Quant_Pagamentos_Via_Boleto': quant_boletos,
                    'Quant_Ocorrencia': quant_ocorrencia,
                }
                ufs_categories = encoder.categories_[0].tolist()

                for feature_name in model.feature_names_in_:
                     if feature_name in processed_numerical_values:
                          input_vector_list.append(processed_numerical_values[feature_name])
                     elif feature_name.startswith('UF_'):
                         try:
                              uf_category = feature_name[3:]
                              if hasattr(encoder, 'drop') and encoder.drop == 'first' and len(ufs_categories) > 0:
                                  categories_after_drop = ufs_categories[1:]
                                  if uf_category in categories_after_drop:
                                       category_index_in_encoded_output = categories_after_drop.index(uf_category)
                                       input_vector_list.append(uf_encoded[0, category_index_in_encoded_output])
                                  else:
                                       input_vector_list.append(0.0)
                              else: # Sem drop='first'
                                  if uf_category in ufs_categories:
                                      category_index_in_original_categories = ufs_categories.index(uf_category)
                                      input_vector_list.append(uf_encoded[0, category_index_in_original_categories])
                                  else:
                                      st.warning(f"Feature '{feature_name}' do modelo não corresponde a uma categoria conhecida pelo encoder. Usando 0.")
                                      input_vector_list.append(0.0)
                         except Exception as e:
                              st.warning(f"Erro ao mapear feature '{feature_name}' para valor encoded na linha {i+2}: {e}. Usando 0.")
                              input_vector_list.append(0.0)
                     else:
                          st.warning(f"Recurso '{feature_name}' do modelo não manipulado pela lógica de processamento. Usando 0.")
                          input_vector_list.append(0.0)

                input_vector = np.array([input_vector_list])
                input_df = pd.DataFrame(input_vector, columns=model.feature_names_in_)

                prob = model.predict_proba(input_df)[:, 1][0]
                prob_value = f"{prob:.4f}".replace('.', ',')

        except Exception as e:
            st.error(f"Erro inesperado no processamento da linha {i+2}: {e}")
            row_error = True

        if row_error:
            prob_value = ''
            errored_rows_count += 1

        updated_values.append(row_processed + [prob_value])
        processed_rows_count += 1

        if len(data_rows) > 0:
             progress_bar.progress((i + 1) / len(data_rows))


    progress_bar.empty()

    if errored_rows_count > 0:
        st.warning(f"Processamento concluído. {processed_rows_count} linhas processadas, com {errored_rows_count} erros.")
    else:
        st.success(f"Processamento concluído. {processed_rows_count} linhas processadas com sucesso.")

    return updated_values


def update_sheet(sheet_id, updated_data):
    """Updates the Google Sheet with the processed data."""
    if not updated_data or (len(updated_data) == 1 and len(updated_data[0]) < 2):
         st.warning("Nenhum dado ou cabeçalho completo para atualizar.")
         return

    service = get_google_sheets_service()

    # Calcular o Range de Atualização com base nos updated_data e no nome da aba
    num_rows_to_update = len(updated_data)
    num_cols_to_update = len(updated_data[0])

    update_last_col_letter = ''
    temp_index = num_cols_to_update
    if temp_index == 0:
         update_last_col_letter = 'A'
    else:
        while temp_index > 0:
            remainder = (temp_index - 1) % 26
            update_last_col_letter = chr(ord('A') + remainder) + update_last_col_letter
            temp_index = (temp_index - 1) // 26

    sheet_name_from_range = RANGE_NAME.split('!')[0] if '!' in RANGE_NAME else 'Página1'
    update_range_string = f'{sheet_name_from_range}!A1:{update_last_col_letter}{num_rows_to_update}'


    st.info(f"Atualizando planilha: {sheet_id} - {update_range_string}")

    try:
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=update_range_string,
            valueInputOption='RAW',
            body={'values': updated_data}
        ).execute()
        st.success('Planilha atualizada com sucesso!')
    except Exception as e:
        st.error(f"Erro ao atualizar planilha: {e}")


# --- Streamlit App Layout ---

st.title("App para Prever Probabilidade de Pagamento Automática")

# --- Lógica de Acionamento Baseada em Parâmetros de URL ---
query_params = st.query_params
print(f'Parâmetros de URL: {query_params}') # Para depuração, pode ser removido depois
# Adiciona um cabeçalho para mostrar os parâmetros de URL

# Verifica se o parâmetro 'trigger' está presente na URL e é 'true'
# A URL vinda do script Apps Script deve ser algo como:
# https://<sua_url_streamlit>/?trigger=true

if 'trigger' in query_params and query_params['trigger'] == 'true':
    # --- Execução Automática ---
    st.header("Processamento Automático Iniciado")
    st.write("Acionado pela planilha Google Sheets.")

    # Adiciona um spinner durante o processamento
    with st.spinner("Processando dados da planilha..."):
        st.markdown("---")
        st.header("Status do Processamento")

        # Executa as funções de processamento e atualização usando as constantes fixas
        updated_data = process_sheet_data(SPREADSHEET_ID, RANGE_NAME)

        # Só atualiza se o processamento não falhou criticamente (retornou None)
        if updated_data is not None:
             update_sheet(SPREADSHEET_ID, updated_data)
        else:
             st.error("Processamento falhou devido a um erro na leitura ou validação inicial dos dados.")

    # Opcional: Adicionar uma mensagem final após a conclusão do processamento
    # st.write("Processamento concluído.")

else:
    # --- Modo de Espera (não acionado) ---
    st.header("Aguardando Acionamento")
    st.info("Este aplicativo está configurado para ser acionado diretamente da sua planilha Google Sheets.")
    st.write("Por favor, use o botão ou menu na planilha para iniciar o processamento automático.")
    st.write(f"Aguardando parâmetro de URL 'trigger=true'. Parâmetros atuais: {dict(query_params)}")


st.markdown("---")
st.write("Desenvolvido com ❤️ e Streamlit.")