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


VECTOR_INDEX_NAME = "chunk_bert_base_uncased_index"  # Example name for the index


def semantic_search(question, model_name, top_k=5):
    """Search for similar nodes using the Neo4j vector index with local embeddings from the specified model."""

    # Retrieve the model and tokenizer from the MODELS dictionary
    model = MODELS[model_name]["model"]
    tokenizer = MODELS[model_name]["tokenizer"]

    # Step 1: Generate the embedding for the query using the specified model
    question_embedding = get_embeddings(question, model, tokenizer)
    # Ensure it's a 1D list for Neo4j query
    question_embedding = question_embedding.flatten().tolist()

    # Step 2: Neo4j query to find similar nodes using the vector search
    vector_search_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $question_embedding) 
        YIELD node, score
        RETURN score, node.text AS text
        ORDER BY score DESC
        LIMIT $top_k
    """

    # Step 3: Execute the query with the generated embedding
    similar = kg.query(vector_search_query,
                       params={
                           # Embedding converted to list for Neo4j
                           'question_embedding': question_embedding,
                           'index_name': VECTOR_INDEX_NAME,  # Name of the vector index in Neo4j
                           'top_k': top_k  # Number of similar results to return
                       })

    # Step 4: Return the results from the query
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
    query = "In a single sentence, tell me about Netapp."
    model_name = "bert_base_uncased"
    results = semantic_search(query, model_name)
    print("Semantic Search Results:")
    # Debugging step: print the result structure
for result in results:
    print("Result keys:", result.keys())  # Print the keys of each result

    # Adjust to reflect the actual keys in the results
    print(f"Score: {result['score']}, Text: {result['text']}")
