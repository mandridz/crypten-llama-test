import pandas as pd
import plotly.graph_objects as go
from flask import Flask, render_template_string

# Input file names with results
input_file1 = "inference_results.txt"
input_file2 = "inference_crypten_result.txt"

# Create a Flask application
app = Flask(__name__)


@app.route('/')
def index():
    # Read data from both files
    data1 = read_data_from_file(input_file1)
    data2 = read_data_from_file(input_file2)

    # Combine data from both files
    inference_time = data1['Inference Time'] + data2['Inference Time']
    num_generated_tokens = data1['Number of Generated Tokens'] + data2['Number of Generated Tokens']
    memory_used = data1['Memory Used'] + data2['Memory Used']

    # Create figures
    fig_inference_time = go.Figure()
    fig_num_generated_tokens = go.Figure()
    fig_memory_used = go.Figure()

    # Add trace for inference time
    fig_inference_time.add_trace(go.Bar(
        x=['Inference Time'],
        y=[inference_time],
        marker_color=['indianred', 'indianred']
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
        marker_color=['lightsalmon', 'lightsalmon']
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
        marker_color=['gold', 'gold']
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


def read_data_from_file(file_name):
    data = {}
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Inference Time:" in line:
                data['Inference Time'] = float(line.split(":")[1].strip().split()[0])
            elif "Number of Generated Tokens:" in line:
                data['Number of Generated Tokens'] = int(line.split(":")[1].strip())
            elif "Memory Used:" in line:
                data['Memory Used'] = float(line.split(":")[1].strip().split()[0])
    return data


if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
