- The document download related part of the project

We have a product documentation written in hungarian on a Confluence page. We assume that we will have the confluence related API key. Using this API key we will locally download the related files into (hopefully) markdown format (if it is not possible then we will have to programaticaly convert these files). It would be very important that the folder relationships are mapped as the structure of the original Confluence pages

we will need pydantic, pydantic settings, and jira python API here

- The RAG related part of the project

The goal of the project is that we will compare the quality of multiple open source embedding models for a hungarian document processing, chunking and retrieval. 

The stack we will use:

dotenv related stuff:
python-dotenv
pydantic settings

RAG related stuff
llamaindex
ragas for evaluating the appropriate metrics for the different models

Models that we want to evaluate (local hostable models only)
Harrier-OSS-v1
Qwen3-Embedding-8B
BGE-M3

Schema handling and validation:
Pydantic

For clear communication, logging and terminal interface related stuff:
tqdm
rich

For code quality validation and formatting:
mypy
ruff
isort
black

Data storage
postgresql with pgvector