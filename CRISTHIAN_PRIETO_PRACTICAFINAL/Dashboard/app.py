import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import os

# 1. Configuración y Carga de Datos Segura
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, 'california_housing.csv')

app = dash.Dash(__name__)

# Intentamos cargar los datos
try:
    df = pd.read_csv(csv_path)
    # Limpieza rápida para mejorar visualización
    df = df[df['AveRooms'] < df['AveRooms'].quantile(0.98)]
except Exception as e:
    print(f"ERROR CRÍTICO: No se encontró el archivo CSV en {csv_path}")
    df = pd.DataFrame() # DataFrame vacío para evitar que colapse

# 2. Diseño de la Interfaz (Layout)
app.layout = html.Div(style={'backgroundColor': '#f9f9f9', 'padding': '40px', 'fontFamily': 'Segoe UI'}, children=[
    
    html.Div(style={'textAlign': 'center', 'marginBottom': '50px'}, children=[
        html.H1("Dashboard Inmobiliario: California Housing", style={'color': '#2c3e50', 'fontSize': '36px'}),
        html.P("Exploración interactiva de la relación entre variables socioeconómicas y el tamaño de la vivienda.",
               style={'color': '#7f8c8d', 'fontSize': '18px'})
    ]),

    html.Div(style={'display': 'flex', 'gap': '20px', 'justifyContent': 'center'}, children=[
        # Panel de Control (Izquierda)
        html.Div(style={'width': '30%', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '15px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}, children=[
            html.H4("Configuración del Gráfico", style={'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
            html.Label("Variable Predictora (X):", style={'fontWeight': 'bold', 'display': 'block', 'marginTop': '20px'}),
            dcc.Dropdown(
                id='variable-selector',
                options=[
                    {'label': 'Ingreso Medio (MedInc)', 'value': 'MedInc'},
                    {'label': 'Edad de la Casa (HouseAge)', 'value': 'HouseAge'},
                    {'label': 'Población (Population)', 'value': 'Population'}
                ],
                value='MedInc',
                clearable=False,
                style={'marginTop': '10px'}
            ),
            html.Hr(),
            html.Div(id='metricas-display', style={'marginTop': '20px'})
        ]),

        # Panel del Gráfico (Derecha)
        html.Div(style={'width': '65%', 'backgroundColor': 'white', 'padding': '10px', 'borderRadius': '15px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}, children=[
            dcc.Graph(id='main-scatter-plot')
        ])
    ])
])

# 3. Lógica Interactiva (Callbacks)
@app.callback(
    [Output('main-scatter-plot', 'figure'),
     Output('metricas-display', 'children')],
    [Input('variable-selector', 'value')]
)
def update_analysis(selected_var):
    if df.empty:
        return px.scatter(title="Error: Datos no cargados"), "No hay datos disponibles"

    # Preparar datos para Regresión Simple
    X = df[[selected_var]].values
    y = df['AveRooms'].values
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)

    # Crear Gráfico (Muestreo para velocidad)
    df_sample = df.sample(n=min(3000, len(df)), random_state=42)
    fig = px.scatter(
        df_sample, x=selected_var, y='AveRooms', 
        trendline="ols",
        title=f"Tendencia: {selected_var} vs Promedio de Habitaciones",
        template="plotly_white",
        color_discrete_sequence=['#3498db']
    )
    
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 40, 'r': 0}, hovermode='closest')

    # Contenido de Métricas Narrativas
    metrics_content = html.Div([
        html.H5("Análisis del Modelo", style={'color': '#2980b9'}),
        html.P([html.B("R² (Precisión): "), f"{r2:.4f}"]),
        html.P([html.B("Pendiente: "), f"{model.coef_[0]:.4f}"]),
        html.P("Interpretación:", style={'fontWeight': 'bold', 'marginTop': '10px'}),
        html.Small(f"El modelo indica que por cada unidad de cambio en {selected_var}, el tamaño de la casa varía en promedio {model.coef_[0]:.2f} habitaciones.")
    ])

    return fig, metrics_content

# 4. Ejecución del Servidor
if __name__ == '__main__':
    app.run(debug=True, port=8050, host='0.0.0.0')