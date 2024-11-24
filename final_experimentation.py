import torch
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from langchain_community.graphs import Neo4jGraph
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'

VECTOR_NODE_LABEL = 'Chunk'
VECTOR_SOURCE_PROPERTY = 'text'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models and tokenizers
MODELS = {
    "bert_base_uncased": {
        "model": BertModel.from_pretrained('bert-base-uncased').to(device),
        "tokenizer": BertTokenizer.from_pretrained('bert-base-uncased')
    }
}

# Neo4j connection
kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=200, length_function=len)

# Step 1: Process JSON File and Split Text into Chunks


def split_form10k_data_from_file(file):
    chunks_with_metadata = []
    file_as_object = json.load(open(file))
    for item in ['item1', 'item1a', 'item7', 'item7a']:
        print(f'Processing {item} from {file}')
        item_text = file_as_object[item]
        item_text_chunks = text_splitter.split_text(item_text)
        chunk_seq_id = 0
        for chunk in item_text_chunks[:20]:
            form_id = file[file.rindex('/') + 1:file.rindex('.')]
            chunks_with_metadata.append({
                'text': chunk,
                'f10kItem': item,
                'chunkSeqId': chunk_seq_id,
                'formId': f'{form_id}',
                'chunkId': f'{form_id}-{item}-chunk{chunk_seq_id:04d}',
                'names': file_as_object['names'],
                'cik': file_as_object['cik'],
                'cusip6': file_as_object['cusip6'],
                'source': file_as_object['source'],
            })
            chunk_seq_id += 1
        print(f'\tSplit into {chunk_seq_id} chunks')
    return chunks_with_metadata

# Step 2: Preprocess Text


def preprocess_text(text, tokenizer=None):
    if tokenizer:
        tokens = tokenizer(text, padding='max_length',
                           truncation=True, return_tensors='pt')
        return tokens["input_ids"]
    return text

# Step 3: Generate Embeddings


def get_embeddings(text, model, tokenizer, device=device):
    # Tokenize the input text and preprocess
    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, padding=True).to(device)

    # Get the model's outputs in a no-gradient context
    with torch.no_grad():
        outputs = model(**inputs)

    # Take the mean of the last hidden state across the sequence (dim=1) for a fixed-size representation
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    return embeddings

# Step 4: Save Chunks and Metadata to Neo4j


def save_chunks_to_neo4j(chunks):
    for chunk in chunks:
        chunk_id = chunk["chunkId"]
        chunk_text = chunk["text"]
        kg.query(f"""
            MERGE (chunk:{VECTOR_NODE_LABEL} {{chunkId: $chunk_id}})
            SET chunk += $properties
        """, params={"chunk_id": chunk_id, "properties": chunk})

# Step 5: Update Embeddings in Neo4j


def update_embeddings_in_neo4j(chunk_id, embedding, model_name):
    embedding_property_name = f"textEmbedding_{model_name}"
    flat_embedding = np.array(embedding).flatten().tolist()
    kg.query(f"""
        MATCH (chunk:{VECTOR_NODE_LABEL} {{chunkId: $chunk_id}})
        SET chunk.{embedding_property_name} = $embedding
    """, params={"chunk_id": chunk_id, "embedding": flat_embedding})

# Step 6: Create Vector Indices


def create_vector_indices(model_name, dimensions):
    embedding_property_name = f"textEmbedding_{model_name}"
    index_name = f"{VECTOR_NODE_LABEL.lower()}_{model_name}_index"
    kg.query(f"""
        CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
        FOR (chunk:{VECTOR_NODE_LABEL})
        ON (chunk.{embedding_property_name})
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: 'cosine'
            }}
        }}
    """)

# Step 7: Perform Semantic Search


def semantic_search(query, model_name, top_k=5):
    """Search for similar nodes using the Neo4j vector index with a specified embedding model."""

    # Embedding property and model loading
    embedding_property_name = f"textEmbedding_{model_name}"
    model = MODELS[model_name]["model"]
    tokenizer = MODELS[model_name]["tokenizer"]

    # Generate the embedding for the query using the model
    query_embedding = get_embeddings(
        query, model, tokenizer).flatten().tolist()

    # Neo4j query to find similar nodes using cosine similarity
    vector_search_query = """
        MATCH (chunk:{VECTOR_NODE_LABEL})
        WHERE chunk.{embedding_property_name} IS NOT NULL
        WITH chunk, gds.similarity.cosine(chunk.{embedding_property_name}, $query_embedding) AS score
        RETURN chunk.chunkId AS id, chunk.{VECTOR_SOURCE_PROPERTY} AS text, score
        ORDER BY score DESC
        LIMIT $top_k
    """

    # Format query string with actual values
    vector_search_query = vector_search_query.format(
        VECTOR_NODE_LABEL=VECTOR_NODE_LABEL,
        embedding_property_name=embedding_property_name,
        VECTOR_SOURCE_PROPERTY=VECTOR_SOURCE_PROPERTY
    )

    # Execute the query
    similar = kg.query(vector_search_query,
                       params={
                           "query_embedding": query_embedding,
                           "top_k": top_k
                       })

    return similar


# Main Pipeline
if __name__ == "__main__":
    # Step 1: Process File
    first_file_name = "./data/form10k/0000950170-23-027948.json"
    chunks = split_form10k_data_from_file(first_file_name)
    save_chunks_to_neo4j(chunks)

    # Step 2: Generate and Store Embeddings
    for model_name, components in MODELS.items():
        model = components["model"]
        tokenizer = components["tokenizer"]

        for chunk in chunks:
            chunk_id = chunk["chunkId"]
            text = chunk["text"]
            embedding = get_embeddings(text, model, tokenizer)
            update_embeddings_in_neo4j(chunk_id, embedding, model_name)

        # Bert model's embedding dimension is hidden_size
        dimensions = model.config.hidden_size
        create_vector_indices(model_name, dimensions)

    # Step 3: Perform Semantic Search
    query = "Risks related to market fluctuations"
    model_name = "bert_base_uncased"
    results = semantic_search(query, model_name)
    print("Semantic Search Results:")
    for result in results:
        print(
            f"ID: {result['id']}, Text: {result['text']}, Score: {result['score']}")
