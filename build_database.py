import os
import chromadb
import pdfplumber
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm  # Import tqdm for progress bars
import sys  # <-- Added this import to fix the NameError
import io  # <-- Added this import as it's used at the bottom

# --- Configuration ---
DB_PATH = "C:/MCP/ragfiles"  # Your Database data path
COLLECTION_NAME = "my_collection_bge3"
MODEL_NAME = 'BAAI/bge-m3'
CHUNK_SIZE = 500
ENCODING_BATCH_SIZE = 32  # Adjust based on your GPU/CPU memory
UPLOAD_BATCH_SIZE = 500  # Adjust based on your system's memory


# --- Text Extraction Functions ---

def extract_text_from_pdf(pdf_path, chunk_size=CHUNK_SIZE):
    """Extracts text from a PDF and splits it into chunks."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    except Exception as e:
        print(f"Warning: Could not process PDF {pdf_path}. Error: {e}")
        return []


def extract_text_from_txt(txt_path, chunk_size=CHUNK_SIZE):
    """Extracts text from a TXT file and splits it into chunks."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    except Exception as e:
        print(f"Warning: Could not process TXT {txt_path}. Error: {e}")
        return []


def extract_text_from_md(md_path, chunk_size=CHUNK_SIZE):
    """Extracts text from a MD file and splits it into chunks."""
    try:
        with open(md_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    except Exception as e:
        print(f"Warning: Could not process MD {md_path}. Error: {e}")
        return []


# --- Main Database Build Logic ---

def build_database():
    """
    Initializes ChromaDB, processes source documents, generates embeddings,
    and populates the database.
    """
    print("--- Starting Database Build Process ---")

    # 1. Connect to ChromaDB
    print(f"Connecting to database at: {DB_PATH}")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    # 2. Safety Check: Ask before overwriting existing data
    existing_collections = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        if collection.count() > 0:
            print(f"\n⚠️  Warning: Collection '{COLLECTION_NAME}' already contains {collection.count()} documents.")
            choice = input("Do you want to delete existing data and rebuild? (y/N): ").lower()
            if choice == 'y':
                print(f"Deleting existing collection '{COLLECTION_NAME}'...")
                chroma_client.delete_collection(name=COLLECTION_NAME)
                collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
                print("Collection deleted. Proceeding with rebuild.")
            else:
                print("Aborting build process. Database remains unchanged.")
                return
    else:
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # 3. Process all PDF, TXT, and MD files in the directory
    print(f"\nProcessing files from directory: {DB_PATH}")
    all_text_chunks = []
    all_document_ids = []

    # Get a list of files to process (now includes .md)
    files_to_process = [f for f in os.listdir(DB_PATH) if f.endswith('.pdf') or f.endswith('.txt') or f.endswith('.md')]
    if not files_to_process:
        print(f"No .pdf, .txt, or .md files found in {DB_PATH}. Exiting.")
        return

    # Process files with a tqdm progress bar
    for filename in tqdm(files_to_process, desc="Processing Files"):
        file_path = os.path.join(DB_PATH, filename)
        chunks = []
        if filename.endswith('.pdf'):
            chunks = extract_text_from_pdf(file_path)
        elif filename.endswith('.txt'):
            chunks = extract_text_from_txt(file_path)
        elif filename.endswith('.md'):
            chunks = extract_text_from_md(file_path)

        start_id = len(all_document_ids)
        all_text_chunks.extend(chunks)
        all_document_ids.extend([f"{filename}_{start_id + i}" for i in range(len(chunks))])

    if not all_text_chunks:
        print("\nNo text chunks were extracted. Make sure your files are not empty or corrupted.")
        return

    print(f"\n✅ Total text chunks created: {len(all_text_chunks)}")

    # 4. Load Embedding Model
    print(f"\nLoading embedding model: {MODEL_NAME} (this may take a moment)...")
    start_time = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

    # 5. Generate Embeddings with Progress Bar
    print("\nGenerating embeddings for all text chunks...")
    all_embeddings = []

    # Process in batches with tqdm
    for i in tqdm(range(0, len(all_text_chunks), ENCODING_BATCH_SIZE), desc="Generating Embeddings"):
        batch_chunks = all_text_chunks[i:i + ENCODING_BATCH_SIZE]
        # Set show_progress_bar=False to avoid nested progress bars
        batch_embeddings = model.encode(batch_chunks, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)

    print(f"✅ Embedding generation complete. Total vectors: {len(all_embeddings)}")

    # 6. Upload data to ChromaDB in batches with Progress Bar
    print("\nUploading documents and embeddings to ChromaDB...")

    # Process in batches with tqdm
    for i in tqdm(range(0, len(all_text_chunks), UPLOAD_BATCH_SIZE), desc="Uploading to ChromaDB"):
        collection.upsert(
            documents=all_text_chunks[i:i + UPLOAD_BATCH_SIZE],
            ids=all_document_ids[i:i + UPLOAD_BATCH_SIZE],
            # Ensure embeddings are in the correct list format
            embeddings=[e.tolist() for e in all_embeddings[i:i + UPLOAD_BATCH_SIZE]]
        )

    print("\n--- ✅ Database Build Complete! ---")
    print(f"Total documents in collection '{COLLECTION_NAME}': {collection.count()}")


if __name__ == "__main__":
    # Ensure stdout supports UTF-8
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            print("Stdout encoding set to UTF-8")
        except Exception as e:
            print(f"Warning: Could not set stdout encoding to UTF-8. Error: {e}")

    build_database()

