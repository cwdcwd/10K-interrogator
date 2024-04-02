

# 10K Interrogator
This project is a Python application that uses OpenAI's GPT-3Turbo model to process and analyze PDF documents. It uses the Elasticsearch service for storing and retrieving data, and the Unstructured.io service for partitioning PDFs into manageable chunks. The application is designed to process PDFs, summarize their content, and answer questions based on the summarized content.

## Introduction
Briefly describe your project here. What does it do? Why is this project useful? Where can users find more information about it? 

## Dependencies
This project requires Python 3.6 or later. The following Python libraries are also required:

- os
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
- 
- [Tesseract OCR](https://tesseract-ocr.github.io/)
- [Poppler PDF rendering library](https://poppler.freedesktop.org/)


## Services
Describe all the services used in your project. 

- OpenAI: This service is used to process and analyze the text data. You will need an API key from OpenAI to use this service.
- Elasticsearch: This service is used for storing and retrieving data. You will need to have an Elasticsearch instance running and accessible to the application.
- Unstructured.io: This service is used to partition PDFs into manageable chunks. You will need an API key from Unstructured.io to use this service.

## Setup
Provide detailed instructions on how to install and run your project locally. 

1. Clone the repository: `git clone https://github.com/yourusername/yourproject.git`
2. Navigate to the project directory: `cd yourproject`
3. Install dependencies: `npm install` or `pip install -r requirements.txt`
4. Run the project: `npm start` or `python manage.py runserver`

## Contributing
Explain how other developers and users can contribute to the project. 

## License
Include a section for the license if applicable. 

## Contact Information
Your contact information so developers can reach out to you. 