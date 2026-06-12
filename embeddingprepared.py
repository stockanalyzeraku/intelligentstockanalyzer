import json

class EmbeddingPrepared:

    MAX_WORDS = 180       # ~roughly under 256 tokens for MiniLM
    OVERLAP_WORDS = 30    # overlap between consecutive chunks


    def split_text_into_chunks(self, text, max_words=MAX_WORDS, overlap=OVERLAP_WORDS):
        """
    Splits text into word-based chunks with overlap.
    Tries to split on paragraph boundaries first; falls back to word-splitting
    if a single paragraph itself exceeds max_words.
    """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        current_words = []

        for para in paragraphs:
            para_words = para.split()

        # If a single paragraph is itself too long, split it by words
            if len(para_words) > max_words:
                # flush current buffer first
                if current_words:
                    chunks.append(' '.join(current_words))
                    current_words = []

                start = 0
                while start < len(para_words):
                    end = start + max_words
                    chunks.append(' '.join(para_words[start:end]))
                    start = end - overlap if end - overlap > start else end
                continue

        # If adding this paragraph would exceed max_words, flush current buffer
            if len(current_words) + len(para_words) > max_words:
                if current_words:
                    chunks.append(' '.join(current_words))
                    # start new buffer with overlap from end of previous chunk
                    overlap_words = current_words[-overlap:] if overlap < len(current_words) else current_words
                    current_words = overlap_words + para_words
                else:
                    current_words = para_words
            else:
                current_words.extend(para_words)

        if current_words:
            chunks.append(' '.join(current_words))

        return chunks


    def prepare_for_embedding(self, input_path, output_path, max_words=MAX_WORDS, overlap=OVERLAP_WORDS):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        records = []
        for page in data:
            metadata = {k: v for k, v in page.items() if k not in ('clean_text', 'raw_tables')}

        # --- Prose: split into chunks if too long ---
            clean_text = page.get("clean_text", "")
            if clean_text.strip():
                text_chunks = self.split_text_into_chunks(clean_text, max_words, overlap)
                for c_idx, chunk in enumerate(text_chunks):
                    records.append({
                        "id": f"page_{page['page_number']}_text_chunk_{c_idx}",
                        "text": chunk,
                        "metadata": metadata
                    })


        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    
        return records

