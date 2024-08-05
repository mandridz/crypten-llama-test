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
    fig_inference_time = go.Figure()
    fig_num_generated_tokens = go.Figure()
    fig_memory_used = go.Figure()

    # Add trace for inference time
    fig_inference_time.add_trace(go.Bar(
        x=['Inference Time'],
        y=[inference_time],
        marker_color='indianred'
    ))
    fig_inference_time.update_layout(
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
    fig_num_generated_tokens.add_trace(go.Bar(
        x=['Number of Generated Tokens'],
        y=[num_generated_tokens],
        marker_color='lightsalmon'
    ))
    fig_num_generated_tokens.update_layout(
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
    fig_memory_used.add_trace(go.Bar(
        x=['Memory Used'],
        y=[memory_used],
        marker_color='gold'
    ))
    fig_memory_used.update_layout(
        title='Memory Usage',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Memory (MB)',
            titlefont_size=16,
            tickfont_size=14,
        ),
        bargap=0.1
    )

    # Generate HTML for the plots
    plot_html_inference_time = fig_inference_time.to_html(full_html=False)
    plot_html_num_generated_tokens = fig_num_generated_tokens.to_html(full_html=False)
    plot_html_memory_used = fig_memory_used.to_html(full_html=False)

    # Render the plots in a simple HTML template
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Inference Results</title>
            <style>
                .container {
                    display: flex;
                    justify-content: space-around;
                    align-items: flex-start;
                }
                .plot {
                    width: 30%; /* Adjust width as needed */
                }
            </style>
        </head>
        <body>
            <h1>Inference Results</h1>
            <div class="container">
                <div class="plot">
                    <h2>Inference Time</h2>
                    {{ plot_html_inference_time|safe }}
                </div>
                <div class="plot">
                    <h2>Number of Generated Tokens</h2>
                    {{ plot_html_num_generated_tokens|safe }}
                </div>
                <div class="plot">
                    <h2>Memory Usage</h2>
                    {{ plot_html_memory_used|safe }}
                </div>
            </div>
        </body>
        </html>
    ''',
    plot_html_inference_time=plot_html_inference_time,
    plot_html_num_generated_tokens=plot_html_num_generated_tokens,
    plot_html_memory_used=plot_html_memory_used)

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)