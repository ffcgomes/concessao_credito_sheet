import os
import numpy as np
import pandas as pd
import joblib
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import streamlit as st
import json

# --- App Configuration ---
st.set_page_config(page_title="Calculadora de Probabilidade de Pagamento")

# --- Hardcoded Google Sheets API Configuration ---
# !!! REPLACE WITH YOUR ACTUAL SPREADSHEET ID !!!
# Example: SPREADSHEET_ID = '1Y0mleR7AdIFsFFgd7Wwt5HhzcdU1DlBTn9LCkYlcAxY'
SPREADSHEET_ID = '1Y0mleR7AdIFsFFgd7Wwt5HhzcdU1DlBTn9LCkYlcAxY' # <<-- COLOQUE O ID DA SUA PLANILHA AQUI

# --- CORREÇÃO: Aumentar o range de LEITURA para incluir a coluna onde a probabilidade será escrita (Coluna G) ---
# Isso evita o erro da API sobre tentar escrever fora do range de leitura.
# Vamos ler até a coluna G para todas as linhas, garantindo que a API "conheça" a coluna G.
RANGE_NAME = 'Página1!A:G' # <<-- LER COLUNAS A ATÉ G em TODAS as linhas. Ajuste 'Página1' se necessário.
# Se você quiser limitar a leitura a um número específico de linhas, use por exemplo:
# RANGE_NAME = 'Página1!A1:G11' # Ler até a linha 11, incluindo a coluna G

SCOPES = ['https://www.googleapis.com/auth/spreadsheets'] # Read and write access

# --- File Paths (adjust as needed for your deployment) ---
MODEL_PATH = "./resultados_parciais/modelo_logistico.pkl" # <<-- Verifique se o caminho do modelo está correto

# --- Streamlit Secrets Key ---
# This key must match the section name in your .streamlit/secrets.toml
GCP_SECRETS_KEY = 'gcp_service_account' # <<-- Verifique se a chave nos secrets.toml é essa

# --- Mappings ---
# Mapping of sheet column names (from the image) to model feature names
# Based on the image, the names are correct, so the mapping is as intended.
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
        # If no values at all, return an empty list.
        return []

    # O cabeçalho é a primeira linha lida
    header_original = values[0]
    # Os dados são as linhas a partir da segunda
    data_rows = values[1:]

    if not data_rows:
         st.warning('Planilha contém apenas o cabeçalho ou está vazia após o cabeçalho no intervalo especificado.')
         # Retorna apenas o cabeçalho lido mais a nova coluna 'Probabilidade'
         return [header_original + ['Probabilidade']]


    # --- Mapear colunas da planilha para nomes de features DO MODELO ---
    # Precisamos encontrar os índices no cabeçalho original APENAS para as colunas
    # que o script lê DIRETAMENTE da planilha ANTES da codificação (numéricas + UF).
    # Não procuramos pelos nomes de features one-hot encoded (UF_AL, etc.) no cabeçalho.

    # Definir as features do modelo que *correspondem diretamente* a colunas na planilha
    # (ignorando as features que virão do One-Hot Encoding)
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
         # Se for, o nome da coluna na planilha é a CHAVE correspondente no mapeamento
         is_mapped = False
         for original_sheet_name, mapped_model_name in COLUMN_MAPPING.items():
              if mapped_model_name == model_feature_name:
                   sheet_col_name_to_find = original_sheet_name # Encontrou o nome da planilha (ex: 'Valor_Parcela')
                   is_mapped = True
                   break

         # Lidar com a coluna 'UF' explicitamente se o nome da feature do modelo for 'UF'
         if model_feature_name == 'UF':
              sheet_col_name_to_find = 'UF' # Garante que procuramos por 'UF' na planilha
              is_mapped = True # Trata 'UF' como mapeada para simplificar a lógica de erro

         # Tentar encontrar o índice desta coluna no cabeçalho lido da planilha
         try:
              idx_map_original[model_feature_name] = header_original.index(sheet_col_name_to_find)
         except ValueError:
              # Se não encontrou a coluna necessária no cabeçalho, mostre um erro detalhado
              error_msg = f"Erro: Coluna esperada '{sheet_col_name_to_find}' não encontrada no cabeçalho da planilha lido."
              if not is_mapped: # Se a feature não estava no mapeamento e não era 'UF', a mensagem já está boa
                   error_msg = f"Erro: Coluna esperada '{model_feature_name}' não encontrada no cabeçalho da planilha lido."

              st.error(error_msg)
              st.info(f"Cabeçalho lido da planilha: {header_original}") # Mostra o cabeçalho lido
              st.info(f"Procurando no cabeçalho por colunas que correspondem a estas features do modelo: {model_features_from_sheet}") # Mostra o que buscava
              return None # Indica falha


    # Prepare list for updating (includes header with new column)
    # Usa o cabeçalho lido da planilha e adiciona a nova coluna 'Probabilidade'
    # Se a coluna 'Probabilidade' já existir no cabeçalho lido (porque o range de leitura agora inclui G),
    # essa linha pode precisar de ajuste para não duplicar, mas a lógica de escrita vai sobrescrever.
    # Uma forma segura: verifica se 'Probabilidade' já existe no cabeçalho lido.
    updated_header = header_original[:] # Copia o cabeçalho lido
    if 'Probabilidade' not in updated_header:
         updated_header.append('Probabilidade')
    # else: # A coluna Probabilidade já estava no cabeçalho lido (e.g., em G1)
         # A lógica de escrita abaixo vai sobrescrever a coluna onde o cabeçalho 'Probabilidade' foi lido.

    updated_values = [updated_header] # Começa a lista de atualização com o cabeçalho


    st.info(f"Processando {len(data_rows)} linhas...")
    progress_bar = st.progress(0)

    processed_rows_count = 0
    errored_rows_count = 0

    # Process each row to calculate probability
    for i, row in enumerate(data_rows):
        # Cria uma cópia da linha lida para processamento
        # Garante que a cópia tenha o mesmo número de colunas do cabeçalho lido
        row_processed = row[:]
        while len(row_processed) < len(header_original): # Usa o header_original para saber quantas colunas foram lidas
            row_processed.append('')

        prob_value = '' # Valor padrão da probabilidade em caso de erro
        row_error = False # Flag para erros na linha

        # --- Ler e Validar Dados da Linha usando os índices encontrados ---
        input_data_for_encoding = {} # Dicionário para armazenar os dados lidos prontos para conversão/codificação

        try:
            # Percorrer APENAS as features que lemos DIRETAMENTE da planilha (numéricas + UF)
            for model_feature_name in model_features_from_sheet:
                 original_idx = idx_map_original[model_feature_name] # Obter o índice no cabeçalho lido
                 value = str(row_processed[original_idx]).strip() # Ler o valor da célula, garantir que é string e remover espaços

                 if not value:
                      # Se algum valor essencial (numérico ou UF) estiver vazio, marque a linha como erro
                      row_error = True # Marca a linha com erro
                      break # Sai deste loop

                 # Armazenar o valor lido
                 input_data_for_encoding[model_feature_name] = value

            # --- Conversão de Tipos e Codificação (somente se não houver erro e todos os essenciais estiverem presentes) ---
            if not row_error:
                try:
                     # Converter as colunas numéricas
                     # Usar .replace(',', '.') para lidar com vírgulas como separador decimal
                     valor_quitacao = float(input_data_for_encoding['ValorQuitacao'].replace(',', '.'))
                     atraso = float(input_data_for_encoding['Atraso'].replace(',', '.'))
                     quant_boletos = float(input_data_for_encoding['Quant_Pagamentos_Via_Boleto'].replace(',', '.'))
                     quant_ocorrencia = float(input_data_for_encoding['Quant_Ocorrencia'].replace(',', '.'))
                     uf = input_data_for_encoding['UF'] # O valor da UF já foi lido acima

                except ValueError as ve:
                     # Erro na conversão de algum valor numérico
                     st.warning(f"Erro de conversão de tipo na linha {i+2}: {ve}. Verifique se os valores numéricos estão corretos.") # Optional warning
                     row_error = True # Marca a linha com erro

            # Codificar UF usando o encoder (somente se não houver erro ainda)
            if not row_error:
                 uf_for_encoding = np.array([[uf]]) # O encoder espera input 2D

                 try:
                     # O encoder vai transformar o valor da UF em um vetor one-hot encoded
                     # Já removemos .toarray() na correção anterior
                     uf_encoded = encoder.transform(uf_for_encoding) # <-- CORRIGIDO ANTES

                     # Lidar com categorias desconhecidas depende da configuração do encoder (handle_unknown)
                     # Verificar se a UF lida existe nas categorias que o encoder conhece, se a configuração for 'error'
                     if hasattr(encoder, 'handle_unknown') and encoder.handle_unknown == 'error':
                          ufs_categories = encoder.categories_[0].tolist()
                          if uf not in ufs_categories:
                               raise ValueError(f"UF '{uf}' não vista durante o treinamento do encoder.")

                 except Exception as e:
                     st.warning(f"Erro durante a codificação da UF '{uf}' na linha {i+2}: {e}")
                     row_error = True # Marca a linha com erro

            # --- Montar Vetor de Entrada FINAL para o Modelo (somente se não houver erro) ---
            # Este vetor DEVE ter as features EXATAS que o modelo espera (model.feature_names_in_)
            # na ORDEM CORRETA.
            if not row_error:
                input_vector_list = []
                # Dicionário para pegar os valores numéricos lidos/convertidos
                processed_numerical_values = {
                    'ValorQuitacao': valor_quitacao,
                    'Atraso': atraso,
                    'Quant_Pagamentos_Via_Boleto': quant_boletos,
                    'Quant_Ocorrencia': quant_ocorrencia,
                }

                # Obter as categorias originais do encoder para mapear as features UF_XX
                ufs_categories = encoder.categories_[0].tolist()


                for feature_name in model.feature_names_in_:
                     if feature_name in processed_numerical_values: # Adiciona features numéricas lidas
                          input_vector_list.append(processed_numerical_values[feature_name])
                     elif feature_name.startswith('UF_'): # Lida com features de UF one-hot encoded (ex: 'UF_AL')
                         # A feature_name será algo como 'UF_AL', 'UF_CE', etc.
                         # Precisamos encontrar qual coluna do vetor uf_encoded corresponde a ela
                         try:
                              uf_category = feature_name[3:] # Extrai o estado (ex: 'AL')

                              # Precisamos encontrar o índice desta categoria NO VETOR CODIFICADO.
                              # Se o encoder usou drop='first', o vetor codificado não tem a primeira coluna.
                              # A ordem das colunas em uf_encoded (se não for esparso e drop='first')
                              # corresponde à ordem das categorias originais A PARTIR DA SEGUNDA CATEGORIA.
                              if hasattr(encoder, 'drop') and encoder.drop == 'first' and len(ufs_categories) > 0:
                                  # Se drop='first', a primeira categoria foi dropada.
                                  # As colunas codificadas correspondem a ufs_categories[1:], ufs_categories[2:]...
                                  categories_after_drop = ufs_categories[1:]
                                  if uf_category in categories_after_drop:
                                       # Encontra a posição da categoria na lista após o drop
                                       category_index_in_encoded_output = categories_after_drop.index(uf_category)
                                       # Pega o valor 0 ou 1 do vetor uf_encoded na posição correta
                                       input_vector_list.append(uf_encoded[0, category_index_in_encoded_output])
                                  else:
                                       # Se a categoria não está nas categorias após o drop,
                                       # é uma categoria desconhecida ou a dropada. O valor para a feature name é 0.
                                       input_vector_list.append(0.0)

                              else: # Se o encoder NÃO usou drop='first' (ou drop='if_binary')
                                  # As colunas codificadas correspondem à ordem original de ufs_categories.
                                  if uf_category in ufs_categories:
                                      category_index_in_original_categories = ufs_categories.index(uf_category)
                                      input_vector_list.append(uf_encoded[0, category_index_in_original_categories])
                                  else:
                                      # Categoria não encontrada nas categorias originais
                                      st.warning(f"Feature '{feature_name}' do modelo não corresponde a uma categoria conhecida pelo encoder (sem drop). Usando 0.")
                                      input_vector_list.append(0.0)


                         except Exception as e:
                              # Lida com erros inesperados ao tentar mapear feature_name para valor encoded
                              st.warning(f"Erro ao mapear feature '{feature_name}' para valor encoded na linha {i+2}: {e}. Usando 0.")
                              input_vector_list.append(0.0)


                     else:
                          # Se o feature_name do modelo não foi encontrado em nenhuma das categorias acima
                          # (numérica ou UF_XX mapeada)
                          st.warning(f"Recurso '{feature_name}' do modelo não manipulado pela lógica de processamento. Usando 0.")
                          input_vector_list.append(0.0)


                # Final input vector como array numpy
                input_vector = np.array([input_vector_list])

                # Criar DataFrame com os nomes EXATOS das colunas que o modelo espera
                input_df = pd.DataFrame(input_vector, columns=model.feature_names_in_)

                # Calcular probabilidade (somente se não houver erro)
                prob = model.predict_proba(input_df)[:, 1][0] # Probabilidade da classe 1 (Pago)

                # Formatar probabilidade como string com vírgula decimal
                prob_value = f"{prob:.4f}".replace('.', ',')

        except Exception as e:
            # Captura quaisquer outros erros inesperados no processamento da linha
            st.error(f"Erro inesperado no processamento da linha {i+2}: {e}")
            row_error = True # Marca a linha com erro

        # Se a linha teve qualquer erro durante o processamento, a probabilidade fica vazia
        if row_error:
            prob_value = ''
            errored_rows_count += 1 # Incrementa contador de erros

        # Adicionar a linha processada (com valor de probabilidade ou vazio) à lista de atualização
        # Usa a cópia row_processed que já tem o padding inicial
        updated_values.append(row_processed + [prob_value])
        processed_rows_count += 1

        # Atualizar barra de progresso
        if len(data_rows) > 0: # Evitar divisão por zero se data_rows estiver vazio
             progress_bar.progress((i + 1) / len(data_rows))


    progress_bar.empty() # Remove a barra de progresso

    if errored_rows_count > 0:
        st.warning(f"Processamento concluído. {processed_rows_count} linhas processadas, com {errored_rows_count} erros (probabilidade não calculada para essas linhas). Verifique os dados nessas linhas.")
    else:
        st.success(f"Processamento concluído. {processed_rows_count} linhas processadas com sucesso.")


    return updated_values # Retorna os dados prontos para atualizar a planilha


# --- Função para Atualizar Planilha ---
# CORREÇÃO: Esta função agora NÃO precisa do range_to_read para calcular o range de atualização final,
# apenas precisa dos updated_data. O range de atualização é calculado com base NO TAMANHO dos dados
# a serem escritos e no nome da aba (tirado da constante RANGE_NAME).
def update_sheet(sheet_id, updated_data):
    """Updates the Google Sheet with the processed data."""
    if not updated_data: # Check if updated_data is empty (e.g., read failed)
         st.warning("Nenhum dado para atualizar na planilha.")
         return
    # Verifica se updated_data contém pelo menos o cabeçalho e a nova coluna
    if len(updated_data) == 0 or len(updated_data[0]) < (len(COLUMN_MAPPING) + 2) : # +1 for Atraso, +1 for UF, +1 for Cliente, +1 for Probabilidade ~ 6+1
        # Uma verificação mais robusta seria checar se updated_data[0][-1] == 'Probabilidade'
        if len(updated_data) == 1 and updated_data[0][-1] == 'Probabilidade' and len(updated_data[0]) > 1:
             st.info("Apenas o cabeçalho da nova coluna foi preparado para atualização.")
             pass # Pode prosseguir para atualizar apenas o cabeçalho
        else:
             st.warning("Dados insuficientes ou malformados para atualização.")
             return


    service = get_google_sheets_service()

    # --- CORREÇÃO: Calcular o Range de Atualização com base nos updated_data e no nome da aba ---
    # O range de atualização começa sempre em A1.
    # O número de linhas é o total de linhas nos updated_data (incluindo cabeçalho).
    # O número de colunas é o total de colunas na primeira linha dos updated_data (cabeçalho original + 'Probabilidade').

    num_rows_to_update = len(updated_data)
    num_cols_to_update = len(updated_data[0]) # Número de colunas na primeira linha (cabeçalho)

    # Calcular a letra da última coluna (1-based) a partir do número de colunas (1=A, 2=B, ...)
    update_last_col_letter = ''
    temp_index = num_cols_to_update # Começa com a contagem 1-based de colunas
    if temp_index == 0: # Caso estranho onde não há colunas
         update_last_col_letter = 'A' # Default para A
    else:
        while temp_index > 0:
            # Calcula o resto para mapear para A-Z (0-25)
            remainder = (temp_index - 1) % 26
            update_last_col_letter = chr(ord('A') + remainder) + update_last_col_letter
            # Divide para ir para a próxima "casa" na notação de coluna (A, B... Z, AA, AB...)
            temp_index = (temp_index - 1) // 26


    # Obter o nome da aba a partir da constante RANGE_NAME
    sheet_name_from_range = RANGE_NAME.split('!')[0] if '!' in RANGE_NAME else 'Página1'

    # Construir a string final do range de atualização (ex: 'Página1!A1:G11')
    update_range_string = f'{sheet_name_from_range}!A1:{update_last_col_letter}{num_rows_to_update}'

    # --- Fim da CORREÇÃO do cálculo do Range de Atualização ---


    st.info(f"Atualizando planilha: {sheet_id} - {update_range_string}")

    try:
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=update_range_string,
            valueInputOption='RAW', # Tratar input como raw strings (sem parsear datas, números, etc.)
            body={'values': updated_data}
        ).execute()
        st.success('Planilha atualizada com sucesso!')
    except Exception as e:
        st.error(f"Erro ao atualizar planilha: {e}")


# --- Streamlit App Layout ---

st.title("App para Prever Probabilidade de Pagamento")

st.write(f"""
Este aplicativo lê dados de uma planilha Google Sheets (ID: `{SPREADSHEET_ID}`),
utiliza um modelo de Machine Learning para prever a probabilidade de um pagamento ser efetuado
no intervalo fixo `{RANGE_NAME}` e escreve a probabilidade de volta na planilha.
""")

st.warning("Certifique-se de que a conta de serviço configurada nos Streamlit Secrets tem permissão de **Editor** na planilha.")
st.info(f"O aplicativo irá ler o intervalo fixo: **{RANGE_NAME}**. Note que este intervalo agora inclui a coluna onde a probabilidade será escrita para evitar erros da API.")
st.info(f"O aplicativo irá atualizar a planilha a partir da célula A1, incluindo todos os dados lidos e adicionando uma coluna de 'Probabilidade' após a última coluna lida.")


# --- Processing Button ---
#if st.button("Processar Planilha e Calcular Probabilidades"):

# Add a spinner for visual feedback during the entire process
with st.spinner("Processando dados da planilha..."):
    st.markdown("---")
    st.header("Status do Processamento")

    # Run the processing function using the hardcoded RANGE_NAME (que agora inclui a coluna G)
    updated_data = process_sheet_data(SPREADSHEET_ID, RANGE_NAME)

    # Only proceed to update if processing was successful and returned data (not None)
    if updated_data is not None:
            # Chamar a função de atualização sem passar o range de leitura novamente
            update_sheet(SPREADSHEET_ID, updated_data)
    else:
            # Se process_sheet_data retornou None devido a um erro crítico (como cabeçalho não encontrado)
            st.error("Processamento falhou devido a um erro na leitura ou validação inicial dos dados.")


st.markdown("---")
st.write("Desenvolvido com ❤️ e Streamlit.")

# --- Optional: Display model info (for debugging/info) ---
# st.sidebar.header("Informações do Modelo")
# try:
#      model_info, encoder_info = load_model_and_encoder(MODEL_PATH)
#      st.sidebar.write("Recursos esperados pelo modelo:")
#      st.sidebar.write(model_info.feature_names_in_.tolist()) # Show as list
#      st.sidebar.write("Categorias da UF no encoder:")
#      st.sidebar.write(encoder_info.categories_[0].tolist()) # Display UF categories as list
#      st.sidebar.write(f"Encoder handle_unknown: {getattr(encoder, 'handle_unknown', 'N/A')}")
#      st.sidebar.write(f"Encoder drop: {getattr(encoder, 'drop', 'N/A')}")
# except Exception as e:
#      st.sidebar.warning(f"Não foi possível carregar informações do modelo na sidebar: {e}")