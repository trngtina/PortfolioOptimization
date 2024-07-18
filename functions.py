import plotly.express as px

def get_distinct_colors(n):
    # Generate n colors from a specified color scale
    colors = px.colors.sample_colorscale('Portland', [i/n for i in range(n)])
    return colors

print(get_distinct_colors(10))