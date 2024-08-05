import pandas as pd
import plotly.graph_objects as go
from flask import Flask, render_template_string

# Input file name with results
input_file = "inference_results.txt"

# Create a Flask application
app = Flask(__name__)

@app.route('/')
def index():
    # Initialize variables to store metrics
    inference_time = None
    num_generated_tokens = None
    memory_used = None

    # Read data from the file
    with open(input_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Inference Time:" in line:
                inference_time = float(line.split(":")[1].strip().split()[0])
            elif "Number of Generated Tokens:" in line:
                num_generated_tokens = int(line.split(":")[1].strip())
            elif "Memory Used:" in line:
                memory_used = float(line.split(":")[1].strip().split()[0])

    # Check if all metrics were found
    if inference_time is None or num_generated_tokens is None or memory_used is None:
        return "Error: Expected metrics are missing in the data.", 500

    # Create figures
    fig = go.Figure()

    # Add trace for inference time
    fig.add_trace(go.Bar(
        x=['Inference Time'],
        y=[inference_time],
        marker_color='indianred'
    ))

    # Update layout for inference time
    fig.update_layout(
        title='Inference Time',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Time (seconds)',
            titlefont_size=16,
            tickfont_size=14,
        ),
        bargap=0.1
    )

    # Add trace for number of generated tokens
    fig.add_trace(go.Bar(
        x=['Number of Generated Tokens'],
        y=[num_generated_tokens],
        marker_color='lightsalmon'
    ))

    # Update layout for number of generated tokens
    fig.update_layout(
        title='Number of Generated Tokens',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Tokens',
            titlefont_size=16,
            tickfont_size=14,
        ),
        bargap=0.1
    )

    # Add trace for memory usage
    fig.add_trace(go.Bar(
        x=['Memory Used'],
        y=[memory_used],
        marker_color='gold'
    ))

    # Update layout for memory usage
    fig.update_layout(
        title='Memory Usage',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Memory (MB)',
            titlefont_size=16,
            tickfont_size=14,
        ),
        bargap=0.1
    )

    # Generate HTML for the plot
    plot_html = fig.to_html(full_html=False)

    # Render the plot in a simple HTML template
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Inference Results</title>
        </head>
        <body>
            <h1>Inference Results</h1>
            {{ plot_html|safe }}
        </body>
        </html>
    ''', plot_html=plot_html)

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)