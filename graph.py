@app.route('/')
def index():
    # Read data from the file
    data = pd.read_csv(input_file, sep='\t')

    # Debug output to check the contents of the DataFrame
    print(data.head())  # Print the first few rows of the DataFrame
    print(data.columns)  # Print the column names

    # Create figures
    fig = go.Figure()

    # Check if the expected columns exist
    if 'Inference Time' in data.columns and 'Number of Generated Tokens' in data.columns and 'Memory Used' in data.columns:
        # Add trace for inference time
        fig.add_trace(go.Bar(
            x=['Inference Time'],
            y=[data['Inference Time'].values[0]],
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
            y=[data['Number of Generated Tokens'].values[0]],
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
            y=[data['Memory Used'].values[0]],
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
    else:
        return "Error: Expected columns are missing in the data.", 500