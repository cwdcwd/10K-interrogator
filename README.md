

# 10K Interrogator
This project is a Python application that uses OpenAI's GPT-3Turbo model to process and analyze PDF documents. It uses the Elasticsearch service for storing and retrieving data, and the Unstructured.io service for partitioning PDFs into manageable chunks. The application is designed to process PDFs, summarize their content, and answer questions based on the summarized content.

## Introduction
This project is a refinement and expansion on [this very fine post](https://datascience.fm/multi-doc-rag-on-10k-reports/). The original project was a proof of concept that demonstrated how to use OpenAI's GPT-3Turbo model to process and analyze PDF documents. The project used the Elasticsearch service for storing and retrieving data, and the Unstructured.io service for partitioning PDFs into manageable chunks. The application was designed to process PDFs, summarize their content, and answer questions based on the summarized content. I've crudely encapuslated some of the original project's code into functions and have been refacaoring it to be more modular and easier to use. 
The source documents are in the form of 10-K reports, which are annual reports filed by publicly traded companies with the U.S. Securities and Exchange Commission (SEC). The reports contain detailed information about the company's financial performance, business operations, and risks. The goal of the project is to extract key information from the reports and answer questions about the content.
I've added a small repl-like interface to the project to make it easier to interact with the code. There is _lots_ of room for improvement here.

## Dependencies
This project requires Python 3.8 or later. The following Python libraries are also required:

- os
- code
- python-dotenv
- uuid
- pickle
- itertools
- operator
- pydantic
- elasticsearch
- unstructured.partition.pdf
- langchain_openai
- langchain.retrievers
- langchain.storage
- langchain_community.vectorstores.elasticsearch
- langchain_core.prompts
- langchain_core.output_parsers
- langchain_core.runnables
- langchain_core.documents
- langchain.output_parsers
- langchain_core.pydantic_v1

## Additional Requirements
- libmagic
- libxml2
- libxslt
- [Tesseract OCR](https://tesseract-ocr.github.io/)
- [Poppler PDF rendering library](https://poppler.freedesktop.org/)

## Services
Describe all the services used in your project. 

- OpenAI: This service is used to process and analyze the text data. You will need an API key from OpenAI to use this service.
- Elasticsearch: This service is used for storing and retrieving data. You will need to have an Elasticsearch instance running and accessible to the application.
- Unstructured.io: This service is used to partition PDFs into manageable chunks. You will need an API key from Unstructured.io to use this service.

## Setup
1. Environment Variables: Create a `.env` file in the project directory (or export them into your environment) with the following environment variables:
```
OPENAI_API_KEY=your_openai_api_key
API_KEY_UNSTRUCTURED = "your_unstructured_api_key"
API_BASE_URL_UNSTRUCTURED = "your specific unstructured.io base url"
ES_HOST = "your_elasticsearch_host (e.g., localhost)"
ES_PORT = "your_elasticsearch_port (e.g., 9200)"
ES_INDEX = "your_elasticsearch_index"
```
2. Elasticsearch: Make sure you have an Elasticsearch instance running and accessible to the application. You can download Elasticsearch from the [official website](https://www.elastic.co/downloads/elasticsearch).
3. Unstructured.io: Sign up for an account on the [Unstructured.io website](https://unstructured.io/) and obtain an API key.
4. OpenAI: Sign up for an account on the [OpenAI website](https://platform.openai.com/) and obtain an API key.
5. Install Dependencies: Run `pip install -r requirements.txt` to install the required Python libraries.
6. Run the Project: Run `python src/rag.py` to start the application.
7. Execute available functions #TODO-- need to add a help and refine available functionss
8. `quit()` to exit the application.

## Contributing
If you'd like to contribute to this project, please create a pull request with your changes.

## License
This project is licensed under the terms of the GNU GENERAL PUBLIC LICENSE. For more details, see the [LICENSE](LICENSE) file.

## Contact Information
cwd at lazybaer dot com