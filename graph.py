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

    # Create figures
    fig_inference_time = go.Figure()
    fig_num_generated_tokens = go.Figure()
    fig_memory_used = go.Figure()

    # Add trace for inference time
    fig_inference_time.add_trace(go.Bar(
        x=['Inference Results', 'Crypten Results'],
        y=[data1['Inference Time'], data2['Inference Time']],
        marker_color=['indianred', 'lightcoral']
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
        x=['Inference Results', 'Crypten Results'],
        y=[data1['Number of Generated Tokens'], data2['Number of Generated Tokens']],
        marker_color=['lightsalmon', 'lightpink']
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
        x=['Inference Results', 'Crypten Results'],
        y=[data1['Memory Used'], data2['Memory Used']],
        marker_color=['gold', 'lightyellow']
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
    # Use pandas to read the data from the file
    df = pd.read_csv(file_name, sep='\t')
    return {
        'Inference Time': df['Inference Time'].values[0],
        'Number of Generated Tokens': df['Number of Generated Tokens'].values[0],
        'Memory Used': df['Memory Used'].values[0]
    }

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)