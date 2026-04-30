"""Ingestion module placeholder for PDF parsing and semantic chunking."""


from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_academic_paper(text: str) -> list[str]:
    """Custom chunker optimized for standard academic paper structures."""

    # We tell the splitter to prioritize breaking at major section headers first!
    custom_separators = [
        "\nAbstract\n",  # Splits right before the Abstract
        "\nI Introduction",  # Splits at Intro (adjust if it's "I\nIntroduction")
        "\nII ",  # Splits at Section 2
        "\nIII ",  # Splits at Section 3
        "\nIndex Terms:",  # Keeps keywords separate
        "\n\n",  # Fallback: Paragraph breaks
        "\n",  # Fallback: Line breaks
        " ",  # Fallback: Words
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=custom_separators,
        is_separator_regex=False,  # Set to True if you want to use Regex for roman numerals
    )

    return text_splitter.split_text(text)
