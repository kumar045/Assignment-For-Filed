import markdown
from weasyprint import HTML, CSS

def generate_report(markdown_text, output_filename='output.pdf', stylesheet=None):
    """
    Convert Markdown text to a PDF file.
    
    Parameters:
    - markdown_text (str): The Markdown content to convert
    - output_filename (str): Name of the output PDF file (default: 'output.pdf')
    - stylesheet (str, optional): Path to a CSS file for custom styling
    
    Returns:
    - str: Path to the generated PDF file
    """
    # Convert Markdown to HTML
    html = markdown.markdown(
        markdown_text, 
        extensions=[
            'extra',  # Adds support for tables, footnotes, etc.
            'codehilite',  # Syntax highlighting for code blocks
            'toc'  # Table of contents support
        ]
    )
    
    # Wrap HTML in a complete HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Markdown PDF</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                line-height: 1.6; 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 20px;
            }}
            code {{ 
                background-color: #f4f4f4; 
                padding: 2px 4px; 
                border-radius: 4px; 
            }}
            pre {{ 
                background-color: #f4f4f4; 
                padding: 10px; 
                border-radius: 5px; 
                overflow-x: auto; 
            }}
            table {{ 
                border-collapse: collapse; 
                width: 100%; 
                margin-bottom: 20px; 
            }}
            th, td {{ 
                border: 1px solid #ddd; 
                padding: 8px; 
                text-align: left; 
            }}
            th {{ 
                background-color: #f2f2f2; 
            }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """
    
    # Apply custom stylesheet if provided
    css = None
    if stylesheet:
        css = CSS(filename=stylesheet)
    
    # Convert HTML to PDF
    HTML(string=full_html).write_pdf(
        output_filename, 
        stylesheets=[css] if css else None
    )
    