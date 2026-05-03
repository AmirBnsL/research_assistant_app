from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
    except Exception as e:
        print(f"⚠️ Error extracting text from PDF: {e}. Trying to read as raw text.")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            pass
    return text


def chunk_academic_paper(text: str) -> list[str]:
    custom_separators = [
        "\nReferences\n",
        "\nAbstract\n",
        "\nIndex Terms:\n",
        "\nI\nIntroduction",
        "\nII\n",
        "\nIII\n",
        "\nIV\n",
        "\n\n",
        "\n",
        " ",
    ]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=custom_separators,
        is_separator_regex=False,
    )
    return text_splitter.split_text(text)
