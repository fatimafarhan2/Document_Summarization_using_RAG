from typing import List , Dict ,Any
import re 

class ChunkSplit:
    def __init__(self, chunksize, chunkoverlap):
        self.chunk_size = chunksize
        self.chunk_overlap = chunkoverlap

    def _cleaning(self, text):
        # here i remove extra space ,any extra characters other than letters digits and punctuation and remove trailing spaces
        print("Cleaning text...")
        data = re.sub(r'\s+', ' ', text)
        data = re.sub(r'[^\w\s.,!?:;\-()]', '', data)
        return data.strip()

    def _splitsentence(self, text: str) -> List[str]:
        # first words, that end with .!? and are followed by space or end of para
        print("Splitting text into sentences...")
        return re.findall(r'.+?(?:[.!?](?:\s+|$))', text)

    def _overlaptext(self, text: str) -> str:
        # in this step we get the trailing overlapping words for next chunk to retain info
        data = text.split()        
        if len(data) > self.chunk_overlap:
            return " ".join(data[-self.chunk_overlap:])
        else:
            return text

    def chunk_document(self, text: str) -> List[Dict[str, Any]]:
        # in this portion we combine all the functionalities and 
        print("Starting chunking process...")
        text = self._cleaning(text)
        sentences = self._splitsentence(text)
        print(f"Total sentences found: {len(sentences)}")

        chunks = []
        temp = []
        count = 0

        for i, sen in enumerate(sentences):
            # split sentence into words and then count
            words = len(sen.split())
            # print(f"Sentence {i + 1}: {words} words")

            # here we see if i am exceeding chunk size by adding words will increase beyond chunk size 
            if count + words > self.chunk_size:
                if temp:
                    chunk_text = " ".join(temp).strip()
                    chunks.append({
                        'text': chunk_text,
                        'length': count,
                        'chunk_id': len(chunks)
                    })
                    print(f"Chunk {len(chunks)} created with {count} words")

                    # here we create new temp by first getting overlapp words
                    overlap_text = self._overlaptext(chunk_text)
                    temp = [overlap_text, sen]
                    count = len((overlap_text + " " + sen).split())
                else:
                    # in case sentence goes beyond chunk limit we forcefully add it 
                    chunks.append({
                        'text': sen.strip(),
                        'length': words,
                        'chunk_id': len(chunks)
                    })
                    # print(f"Long sentence forced into Chunk {len(chunks)}")
                    temp = []
                    count = 0
            else:
                temp.append(sen)
                count += words

        if temp:
            chunk_text = " ".join(temp).strip()
            chunks.append({
                'text': chunk_text,
                'length': count,
                'chunk_id': len(chunks)
            })
            print(f"Final chunk {len(chunks)} created with {count} words")

        print(f"Total chunks created: {len(chunks)}")
        return chunks
