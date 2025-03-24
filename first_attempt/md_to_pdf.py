import markdown
import fitz  # pymupdf
from pathlib import Path
import os


def convert_md_to_pdf(markdown_text, output_file):
    """Convert a Markdown string to a PDF file using pymupdf."""
    try:
        # Convert Markdown to HTML
        html_content = markdown.markdown(
            markdown_text, extensions=["extra", "toc", "codehilite"]
        )

        # Create a new PDF document
        doc = fitz.open()
        page = doc.new_page()

        # Insert HTML as text into the PDF
        text_rect = fitz.Rect(50, 50, 550, 800)  # Define text box dimensions
        page.insert_textbox(text_rect, html_content, fontsize=12)

        # Save the PDF
        os.makedirs(Path(output_file).parent, exist_ok=True)
        doc.save(output_file)
        doc.close()
        print(f"PDF successfully created: {output_file}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    sample_markdown = """# Sample Title\n\nThis is a **Markdown** to PDF converter.\n\n- Bullet point 1\n- Bullet point 2\n\n```python\nprint('Hello, world!')\n```"""
    output_pdf = "output.pdf"
    convert_md_to_pdf(sample_markdown, output_pdf)
