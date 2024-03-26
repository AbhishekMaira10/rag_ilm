from fastapi import FastAPI
from ragatouille import RAGPretrainedModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI(title="nanonets API", version="0.1.1")


class RetriveRelevantChunks():

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        max_token_support: int
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.max_accepted_chunks = max_token_support // chunk_size
        self.rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    def split_text_into_chunks(self, text: str):
        """
        Split the text into chunks of the specified size and overlap.
        """
        return self.text_splitter.split_text(text)

    def get_relevant_text(self, long_text: str, query: str):
        """
        Get the relevant text from the long text.
        """
        chunks = self.split_text_into_chunks(long_text)
        # encode the chunks
        self.rag.encode(documents=chunks)
        # retrive the relevant chunks
        extracted_docs = self.rag.search_encoded_docs(
            query=query,
            k=self.max_accepted_chunks
        )
        # print(extracted_docs)
        # reorder the chunks the way they were in the original text
        ordered_texts = sorted(
            extracted_docs, key=lambda x: x['rank'], reverse=False)
        # clear the memory
        self.rag.clear_encoded_docs(force=True)
        return '\n-----------------------------------------\n'.join([text["content"] for text in ordered_texts])


rag_data = {
    "07461252_Redacted.pdf": [
        {
            "value": "1/5/2023",
            "page": "31",
        },
        {
            "value": "yes",
            "page": "36",
        },
        {
            "value": "yes",
            "page": "39",
        },
        {
            "value": "8/3/2022",
            "page": "38",
        },
        {
            "value": "yes",
            "page": "1",
        },
        {
            "value": "none",
            "page": "",
        },
        {
            "value": "no",
            "page": "1",
        },
    ],
    "07551349_Redacted.pdf": [
        {
            "value": "1/9/2023",
            "page": "1",
        },
        {
            "value": "yes",
            "page": "1",
        },
        {
            "value": "yes",
            "page": "9",
        },
        {
            "value": "3/13/2023",
            "page": "9",
        },
        {
            "value": "yes",
            "page": "3",
        },
        {
            "value": "12/22/2022",
            "page": "12",
        },
        {
            "value": "yes",
            "page": "12",
        },
    ],
    "11062816_Redacted.pdf": [
        {
            "value": "11/30/2023",
            "page": "1",
        },
        {
            "value": "yes",
            "page": "1",
        },
        {
            "value": "yes",
            "page": "18",
        },
        {
            "value": "12/04/2023",
            "page": "18",
        },
        {
            "value": "yes",
            "page": "12",
        },
        {
            "value": "none",
            "page": "",
        },
        {
            "value": "no",
            "page": "1",
        },
    ],
    "11240898_Redacted.pdf": [
        {
            "value": "9/28/2023",
            "page": "23",
        },
        {
            "value": "yes",
            "page": "24",
        },
        {
            "value": "yes",
            "page": "21",
        },
        {
            "value": "none",
            "page": "",
        },
        {
            "value": "yes",
            "page": "8",
        },
        {
            "value": "none",
            "page": "",
        },
        {
            "value": "no",
            "page": "1",
        },
    ],
    "11223843_Redacted.pdf": [
        {
            "value": "08/20/2023",
            "page": "2",
        },
        {
            "value": "yes",
            "page": "9",
        },
        {
            "value": "yes",
            "page": "23",
        },
        {
            "value": "11/7/2023",
            "page": "23",
        },
        {
            "value": "yes",
            "page": "10",
        },
        {
            "value": "none",
            "page": "",
        },
        {
            "value": "no",
            "page": "1",
        },
    ],
    "11341133_Redacted.pdf": [
        {
            "value": "9/27/2023",
            "page": "33",
        },
        {
            "value": "yes",
            "page": "33",
        },
        {
            "value": "yes",
            "page": "34",
        },
        {
            "value": "none",
            "page": "",
        },
        {
            "value": "yes",
            "page": "2",
        },
        {
            "value": "none",
            "page": "",
        },
        {
            "value": "no",
            "page": "1",
        },
    ],
    "11107694_Redacted.pdf": [
        {
            "value": "12/1/2023",
            "page": "3",
        },
        {
            "value": "yes",
            "page": "9",
        },
        {
            "value": "yes",
            "page": "13",
        },
        {
            "value": "7/7/2023",
            "page": "13",
        },
        {
            "value": "yes",
            "page": "20",
        },
        {
            "value": "none",
            "page": "",
        },
        {
            "value": "no",
            "page": "1",
        },
    ],
    "10995925_Redacted.pdf": [
        {
            "value": "none",
            "page": "",
        },
        {
            "value": "yes",
            "page": "6",
        },
        {
            "value": "yes",
            "page": "55",
        },
        {
            "value": "07/24/2023",
            "page": "55",
        },
        {
            "value": "yes",
            "page": "30",
        },
        {
            "value": "none",
            "page": "",
        },
        {
            "value": "no",
            "page": "1",
        },
    ],
    "07625464_Redacted.pdf": [
        {
            "value": "12/30/2021",
            "page": "1",
        },
        {
            "value": "yes",
            "page": "1",
        },
        {
            "value": "no",
            "page": "1",
        },
        {
            "value": "none",
            "page": "",
        },
        {
            "value": "yes",
            "page": "6",
        },
        {
            "value": "none",
            "page": "",
        },
        {
            "value": "no",
            "page": "1",
        },
    ],
    "10708229_Redacted.pdf": [
        {
            "value": "10/30/2023",
            "page": "1",
        },
        {
            "value": "yes",
            "page": "1",
        },
        {
            "value": "yes",
            "page": "33",
        },
        {
            "value": "11/21/2023",
            "page": "33",
        },
        {
            "value": "yes",
            "page": "16",
        },
        {
            "value": "none",
            "page": "",
        },
        {
            "value": "no",
            "page": "1",
        },
    ],
    "07603879_Redacted.pdf": [
        {
            "value": "8/3/2022",
            "page": "3",
        },
        {
            "value": "yes",
            "page": "3",
        },
        {
            "value": "yes",
            "page": "72",
        },
        {
            "value": "11/30/2021",
            "page": "33",
        },
        {
            "value": "yes",
            "page": "7",
        },
        {
            "value": "none",
            "page": "",
        },
        {
            "value": "no",
            "page": "1",
        },
    ],
}


def normalize_key(filename):
    """
    Normalize the filename to a standard format.
    This example removes any trailing ' (number).pdf' and keeps it as 'filename.pdf'.
    Adjust the logic as needed to match your specific normalization rules.
    """
    import re
    # Regular expression to match ' (number)' before the '.pdf' and remove it
    normalized_filename = re.sub(r" \(\d+\)(?=.pdf$)", "", filename)
    return normalized_filename


@app.get("/hi")
async def hi():
    return {"message": "Hi"}


@app.get("/get-rag-data")
async def get_rag_data(file_name: str):
    normalized_key = normalize_key(file_name)
    file_data = rag_data.get(normalized_key)
    if file_data is None:
        return {"error": "File name not found."}
    return {"data": file_data}
