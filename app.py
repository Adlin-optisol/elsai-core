import streamlit as st
from elsai_ocr_extractors.visionai_extractor import VisionAIExtractor
from elsai_ocr_extractors.awstextract import AwsTextractConnector
from elsai_ocr_extractors.llama_parse_extractor import LlamaParseExtractor
from elsai_ocr_extractors.azure_document_intelligence import AzureDocumentIntelligence
from elsai_ocr_extractors.azure_cognitive_service import AzureCognitiveService
from elsai_prompts.pezzo import PezzoPromptRenderer
from dotenv import load_dotenv
import os
import tempfile

# Load environment variables
load_dotenv()

st.title("ELSAI Core Testing Interface")

# Create tabs
tab2, tab3 = st.tabs(["OCR Extractors", "Prompts"])

with tab2:
    st.header("OCR Extractors")
    
    # OCR Extractor selection
    extractor_type = st.selectbox(
        "Select OCR Extractor",
        ["Vision AI", "AWS Textract", "Llama Parser", "Azure Document Intelligence", "Azure Cognitive"]
    )
    
    # File upload
    uploaded_file = st.file_uploader("Upload a file", type=['pdf', 'csv'])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        if st.button("Extract Text"):
            with st.spinner("Extracting text..."):
                try:
                    if extractor_type == "Vision AI":
                        api_key = st.secrets["VISIONAI_API_KEY"]
                        extractor = VisionAIExtractor(api_key=api_key)
                        text = extractor.extract_text_from_pdf(pdf_path=tmp_file_path)
                        st.write("Extracted Text:")
                        st.write(text)
                    
                    elif extractor_type == "AWS Textract":
                        extractor = AwsTextractConnector(
                            access_key=st.secrets["AWS_ACCESS_KEY_ID"],
                            secret_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
                            session_token=st.secrets["AWS_SESSION_TOKEN"],
                            region_name=st.secrets.get("AWS_REGION", "us-east-1")
                        )
                        text = extractor.extract_text(tmp_file_path)
                        st.write("Extracted Text:")
                        st.write(text)
                    
                    elif extractor_type == "Azure Document Intelligence":
                        extractor = AzureDocumentIntelligence(
                            file_path = tmp_file_path,
                            vision_endpoint=st.secrets["VISION_ENDPOINT"],
                            vision_key=st.secrets["VISION_KEY"]
                        )
                        text = extractor.extract_text()
                        st.write("Extracted Text:")
                        st.write(text)
                    
                    elif extractor_type == "Azure Cognitive":
                        extractor = AzureCognitiveService(
                            file_path = tmp_file_path,
                            endpoint=st.secrets["AZURE_COGNITIVE_SERVICE_ENDPOINT"],
                            subscription_key=st.secrets["AZURE_COGNITIVE_SERVICE_SUBSCRIPTION_KEY"]
                        )
                        text = extractor.extract_text_from_pdf()
                        st.write("Extracted Text:")
                        st.write(text)
                    
                    else:  # Llama Parser
                        api_key = st.secrets["LLAMA_PARSER_API_KEY"]
                        extractor = LlamaParseExtractor(api_key=api_key)
                        if uploaded_file.name.endswith('.csv'):
                            data = extractor.load_csv(tmp_file_path)
                            st.write("CSV Data:")
                            st.write(data)
                        else:
                            st.warning("Llama Parser currently only supports CSV files")
                
                except Exception as e:
                    st.error(f"Error during extraction: {str(e)}")
                
                finally:
                    # Clean up the temporary file
                    os.unlink(tmp_file_path)

with tab3:
    st.header("Prompts")
    
    # Pezzo configuration
    st.subheader("Pezzo Prompt Configuration")
    
    # Input fields for Pezzo configuration
    api_key = st.text_input("Pezzo API Key", value=st.secrets.get("PEZZO_API_KEY", ""), type='password')
    project_id = st.text_input("Project ID", value=st.secrets.get("PEZZO_PROJECT_ID", ""))
    server_url = st.text_input("Server URL", value=st.secrets.get("PEZZO_SERVER_URL", "https://elsai-prompts-proxy.optisolbusiness.com"))
    environment = st.selectbox("Environment", ["Production", "Development"], index=0)
    
    # Prompt name input
    prompt_name = st.text_input("Prompt Name", value="sample")
    
    if st.button("Get Prompt"):
        if api_key and project_id:
            try:
                with st.spinner("Fetching prompt..."):
                    pezzo = PezzoPromptRenderer(
                        api_key=api_key,
                        project_id=project_id,
                        server_url=server_url,
                        environment=environment
                    )
                    prompt = pezzo.get_prompt(prompt_name)
                    st.write("Retrieved Prompt:")
                    st.write(prompt)
            except Exception as e:
                st.error(f"Error fetching prompt: {str(e)}")
        else:
            st.warning("Please provide API Key and Project ID") 