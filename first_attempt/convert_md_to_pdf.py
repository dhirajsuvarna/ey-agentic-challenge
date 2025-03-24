import markdown
from weasyprint import HTML

def convert_md_to_pdf(markdown_text, output_file):
    """Convert Markdown text to a PDF file."""
    try:
        # Convert Markdown to HTML
        html_content = markdown.markdown(markdown_text, extensions=['extra', 'toc', 'codehilite'])
        
        # Generate PDF from HTML
        HTML(string=html_content).write_pdf(output_file)
        print(f"PDF successfully created: {output_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    sample_markdown = """# Sample Title\n\nThis is a **Markdown** to PDF converter.\n\n- Bullet point 1\n- Bullet point 2\n\n```python\nprint('Hello, world!')\n```"""
    output_pdf = "output.pdf"
    convert_md_to_pdf(sample_markdown, output_pdf)
