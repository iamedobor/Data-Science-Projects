# ==============================================================================
# INTERACTIVE DASHBOARD - TRADE AND GDP ANALYSIS
# ==============================================================================

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime

# ==============================================================================
# LOAD DATA
# ==============================================================================

print("📊 Loading data...")
df = pd.read_csv('data/featured_data.csv')

# Load cluster data if available
try:
    df_cluster = pd.read_csv('data/cluster_data.csv')
    has_clusters = True
    print("✅ Cluster data loaded")
except:
    has_clusters = False
    print("⚠️  Cluster data not found (optional)")

# Load summary stats
try:
    with open('data/summary_stats.json', 'r') as f:
        summary_stats = json.load(f)
    print("✅ Summary stats loaded")
except:
    summary_stats = {
        'total_countries': df['Country'].nunique(),
        'total_years': df['Year'].nunique(),
        'total_observations': len(df),
        'year_min': int(df['Year'].min()),
        'year_max': int(df['Year'].max())
    }
    print("⚠️  Summary stats generated from data")

print(f"✅ Data loaded: {summary_stats['total_countries']} countries, "
      f"{summary_stats['year_min']}-{summary_stats['year_max']}")

# ==============================================================================
# THEME CONFIGURATIONS
# ==============================================================================

LIGHT_THEME = {
    'bg_primary': '#FFFFFF',
    'bg_secondary': '#F8F9FA',
    'text_primary': '#212529',
    'text_secondary': '#6C757D',
    'border': '#DEE2E6',
    'accent': '#0D6EFD',
    'success': '#198754',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'plotly_template': 'plotly_white'
}

DARK_THEME = {
    'bg_primary': '#1E1E1E',
    'bg_secondary': '#2D2D2D',
    'text_primary': '#E0E0E0',
    'text_secondary': '#B0B0B0',
    'border': '#404040',
    'accent': '#4A9EFF',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'plotly_template': 'plotly_dark'
}

# ==============================================================================
# INITIALIZE DASH APP
# ==============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="Trade and GDP Analysis Dashboard"
)

server = app.server  # For deployment

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_country_data(country):
    """Get all data for a specific country"""
    return df[df['Country'] == country].sort_values('Year')

def format_number(num, prefix='', suffix='', decimals=2):
    """Format large numbers with K, M, B, T suffixes"""
    if pd.isna(num):
        return 'N/A'
    
    abs_num = abs(num)
    if abs_num >= 1e12:
        return f"{prefix}{num/1e12:.{decimals}f}T{suffix}"
    elif abs_num >= 1e9:
        return f"{prefix}{num/1e9:.{decimals}f}B{suffix}"
    elif abs_num >= 1e6:
        return f"{prefix}{num/1e6:.{decimals}f}M{suffix}"
    elif abs_num >= 1e3:
        return f"{prefix}{num/1e3:.{decimals}f}K{suffix}"
    else:
        return f"{prefix}{num:.{decimals}f}{suffix}"

def apply_country_region_filter(selected_countries, region_filter):
    """
    Apply country and region filters with hierarchy:
    1. First apply country filter (all or specific)
    2. Then narrow down by region if specified
    """
    # Step 1: Get countries from country dropdown
    if 'all' in selected_countries:
        countries = df['Country'].unique().tolist()
    else:
        countries = selected_countries
    
    # Step 2: Apply region filter to narrow down
    if region_filter != 'all':
        # Get countries in the selected region
        region_countries = df[df['Region'] == region_filter]['Country'].unique().tolist()
        # Intersect with selected countries
        countries = [c for c in countries if c in region_countries]
    
    return countries

def create_metric_card(title, value, change=None, icon="fas fa-chart-line", theme=LIGHT_THEME):
    """Create a metric card component"""
    change_badge = ""
    if change is not None:
        color = theme['success'] if change >= 0 else theme['danger']
        arrow = "↑" if change >= 0 else "↓"
        change_badge = html.Span(
            f"{arrow} {abs(change):.1f}%",
            style={'color': color, 'fontSize': '0.9rem', 'marginLeft': '10px'}
        )
    
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.I(className=icon, style={'fontSize': '1.5rem', 'color': theme['accent']}),
                html.H6(title, style={'marginTop': '10px', 'color': theme['text_secondary']}),
                html.H3(value, style={'marginTop': '5px', 'color': theme['text_primary']}),
                change_badge
            ])
        ]),
        style={
            'backgroundColor': theme['bg_secondary'],
            'border': f"1px solid {theme['border']}",
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }
    )

# ==============================================================================
# LAYOUT COMPONENTS
# ==============================================================================

def create_navbar(theme):
    """Create navigation bar"""
    return dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.I(className="fas fa-globe", style={'fontSize': '2rem', 'marginRight': '15px'})),
                    dbc.Col(dbc.NavbarBrand("Trade and GDP Analysis Dashboard", className="ms-2", 
                                           style={'fontSize': '1.5rem', 'fontWeight': 'bold'})),
                ], align="center", className="g-0"),
                href="/",
                style={"textDecoration": "none", 'color': theme['text_primary']}
            ),
            dbc.Row([
                dbc.Col([
                    # PDF Export Button
                    dbc.Button(
                        [html.I(className="fas fa-file-pdf"), " Generate Report"],
                        id="export-pdf-btn",
                        style={
                            'backgroundColor': '#DC3545',
                            'color': '#FFFFFF',
                            'border': 'none',
                            'fontWeight': 'bold',
                            'marginRight': '10px'
                        },
                        size="sm"
                    ),
                    # Theme Toggle Button
                    dbc.Button(
                        [html.I(className="fas fa-moon", id="theme-icon"), " Toggle Theme"],
                        id="theme-toggle",
                        style={
                            'backgroundColor': '#FFFFFF',
                            'color': '#000000',
                            'border': '1px solid #CCCCCC',
                            'fontWeight': 'bold'
                        },
                        size="sm"
                    )
                ])
            ], className="ms-auto flex-nowrap mt-3 mt-md-0", align="center")
        ], fluid=True),
        style={'backgroundColor': theme['bg_primary'], 'borderBottom': f"2px solid {theme['border']}"},
        sticky="top"
    )

def create_sidebar(theme):
    """Create sidebar with filters"""
    countries = sorted(df['Country'].unique())
    regions = sorted(df['Region'].unique())
    years = sorted(df['Year'].unique())
    
    return html.Div([
        html.H5("📊 Filters and Controls", style={'marginBottom': '20px', 'color': theme['text_primary']}),
        
        html.Hr(style={'borderColor': theme['border']}),
        
        # Country Selection
        html.Label("Select Countries:", style={'fontWeight': 'bold', 'color': theme['text_primary']}),
        dcc.Dropdown(
            id='country-dropdown',
            options=[{'label': 'All Countries (220)', 'value': 'all'}] + 
                    [{'label': country, 'value': country} for country in countries],
            value=['all'],
            multi=True,
            placeholder="Select countries...",
            style={'marginBottom': '20px'}
        ),
        
        # Region Filter
        html.Label("Filter by Region:", style={'fontWeight': 'bold', 'color': theme['text_primary']}),
        dcc.Dropdown(
            id='region-dropdown',
            options=[{'label': 'All Regions', 'value': 'all'}] + 
                    [{'label': region, 'value': region} for region in regions],
            value='all',
            placeholder="Select region...",
            style={'marginBottom': '20px'}
        ),
        
        # Specific Year Filter
        html.Label("Select Specific Year:", style={'fontWeight': 'bold', 'color': theme['text_primary'], 'marginTop': '10px'}),
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': 'All Years (1970-2021)', 'value': 'all'}] + 
                    [{'label': str(int(year)), 'value': int(year)} for year in years],
            value='all',
            placeholder="Select year...",
            style={'marginBottom': '20px'}
        ),
        
        # Year Range Slider
        html.Label("Year Range:", style={'fontWeight': 'bold', 'color': theme['text_primary']}),
        dcc.RangeSlider(
            id='year-slider',
            min=min(years),
            max=max(years),
            value=[min(years), max(years)],
            marks={int(year): str(int(year)) if year % 10 == 0 else '' for year in years},
            tooltip={"placement": "bottom", "always_visible": True},
            step=1
        ),
        
        html.Hr(style={'borderColor': theme['border'], 'marginTop': '30px'}),
        
        # Quick Stats
        html.H6("📈 Dataset Overview", style={'marginTop': '20px', 'color': theme['text_primary']}),
        html.Div([
            html.P([html.Strong("Countries: "), f"{summary_stats['total_countries']}"], 
                   style={'color': theme['text_secondary']}),
            html.P([html.Strong("Years: "), f"{summary_stats['year_min']}-{summary_stats['year_max']}"], 
                   style={'color': theme['text_secondary']}),
            html.P([
                html.Strong("Observations: "), 
                f"{summary_stats['total_observations']:,}",
                html.Span(" ℹ️", id="obs-tooltip", style={'cursor': 'pointer', 'marginLeft': '5px'})
            ], style={'color': theme['text_secondary']}),
            dbc.Tooltip(
                "Observations = Country-Year combinations. For example: USA in 2020 = 1 observation.",
                target="obs-tooltip",
                placement="right"
            )
        ]),
        
    ], style={
        'padding': '20px',
        'backgroundColor': theme['bg_secondary'],
        'height': '100vh',
        'position': 'fixed',
        'overflowY': 'auto',
        'borderRight': f"1px solid {theme['border']}"
    })

def create_main_content(theme):
    """Create main content area with tabs"""
    tabs = [
        dbc.Tab(label="🌍 Overview", tab_id="overview"),
        dbc.Tab(label="📈 Time Series", tab_id="timeseries"),
        dbc.Tab(label="🔗 Correlation", tab_id="correlation"),
        dbc.Tab(label="🏆 Rankings", tab_id="rankings"),
        dbc.Tab(label="🌐 Map", tab_id="map"),
        dbc.Tab(label="📚 Glossary", tab_id="glossary"),
    ]
    
    if has_clusters:
        tabs.insert(5, dbc.Tab(label="📊 Clusters", tab_id="clusters"))
    
    return html.Div([
        dbc.Tabs(tabs, id="tabs", active_tab="overview", style={'marginBottom': '20px'}),
        html.Div(id="tab-content", style={'padding': '20px'})
    ], style={'padding': '20px'})

# ==============================================================================
# TAB CONTENT CREATORS
# ==============================================================================

def create_overview_tab(selected_countries, region_filter, year_filter, year_range, theme):
    """Overview tab with key metrics"""
    # Apply country and region filters
    selected_countries = apply_country_region_filter(selected_countries, region_filter)
    
    if not selected_countries:
        return html.Div("No countries match the selected filters", 
                       style={'textAlign': 'center', 'padding': '50px', 'color': theme['text_secondary']})
    
    # Filter data based on year selection
    if year_filter == 'all':
        filtered_df = df[
            (df['Country'].isin(selected_countries)) &
            (df['Year'] >= year_range[0]) &
            (df['Year'] <= year_range[1])
        ]
        title_suffix = f"(Total {year_range[0]}-{year_range[1]})"
    else:
        filtered_df = df[
            (df['Country'].isin(selected_countries)) &
            (df['Year'] == year_filter)
        ]
        title_suffix = f"({year_filter})"
    
    # Calculate aggregate metrics - SUM for totals, MEAN for ratios
    total_gdp = filtered_df['Gross Domestic Product (GDP)'].sum()
    total_trade = filtered_df['Total_Trade'].sum()
    avg_openness = filtered_df['Trade_Openness'].mean()
    avg_gdp_pc = filtered_df['GDP_per_Capita'].mean()
    
    # Metric cards
    metrics_row = dbc.Row([
        dbc.Col(create_metric_card(
            "Total GDP", 
            format_number(total_gdp, prefix='$', decimals=2),
            icon="fas fa-dollar-sign",
            theme=theme
        ), width=3),
        dbc.Col(create_metric_card(
            "Total Trade", 
            format_number(total_trade, prefix='$', decimals=2),
            icon="fas fa-exchange-alt",
            theme=theme
        ), width=3),
        dbc.Col(create_metric_card(
            "Avg Trade Openness", 
            f"{avg_openness:.1f}%",
            icon="fas fa-chart-pie",
            theme=theme
        ), width=3),
        dbc.Col(create_metric_card(
            "Avg GDP per Capita", 
            format_number(avg_gdp_pc, prefix='$', decimals=0),
            icon="fas fa-user",
            theme=theme
        ), width=3),
    ], className="mb-4")
    
    # Country summary table
    if year_filter == 'all':
        comparison_df = filtered_df.groupby('Country').agg({
            'Gross Domestic Product (GDP)': 'sum',
            'Total_Trade': 'sum',
            'Trade_Openness': 'mean',
            'GDP_per_Capita': 'mean',
            'Net_Exporter': lambda x: x.sum() / len(x) * 100
        }).reset_index()
        comparison_df['Net_Exporter'] = comparison_df['Net_Exporter'].round(1)
    else:
        comparison_df = filtered_df[[
            'Country', 'Gross Domestic Product (GDP)', 'Total_Trade', 
            'Trade_Openness', 'GDP_per_Capita', 'Net_Exporter'
        ]].copy()
    
    # Sort by GDP descending
    comparison_df = comparison_df.sort_values('Gross Domestic Product (GDP)', ascending=False)
    
    # Format table with $ and commas
    comparison_df['GDP (Billions)'] = comparison_df['Gross Domestic Product (GDP)'].apply(
        lambda x: f"${x/1e9:,.2f}B"
    )
    comparison_df['Trade (Billions)'] = comparison_df['Total_Trade'].apply(
        lambda x: f"${x/1e9:,.2f}B"
    )
    comparison_df['Trade_Openness'] = comparison_df['Trade_Openness'].apply(
        lambda x: f"{x:.1f}%"
    )
    comparison_df['GDP_per_Capita'] = comparison_df['GDP_per_Capita'].apply(
        lambda x: f"${x:,.0f}"
    )
    
    if year_filter == 'all':
        comparison_df['Trade Status'] = comparison_df['Net_Exporter'].apply(
            lambda x: f"🟢 {x:.1f}% Exporter" if x > 50 else f"🔴 {100-x:.1f}% Importer"
        )
    else:
        comparison_df['Trade Status'] = comparison_df['Net_Exporter'].map({True: '🟢 Exporter', False: '🔴 Importer'})
    
    display_df = comparison_df[[
        'Country', 'GDP (Billions)', 'Trade (Billions)', 
        'Trade_Openness', 'GDP_per_Capita', 'Trade Status'
    ]]
    
    # Create table with max height for scrolling when many countries
    table_style = {'color': theme['text_primary']}
    if len(display_df) > 20:
        table_container_style = {
            'maxHeight': '600px',
            'overflowY': 'auto',
            'display': 'block'
        }
    else:
        table_container_style = {}
    
    table = html.Div([
        dbc.Table.from_dataframe(
            display_df, 
            striped=True, 
            bordered=True, 
            hover=True,
            style=table_style
        )
    ], style=table_container_style)
    
    # Region indicator
    region_indicator = ""
    if region_filter != 'all':
        region_indicator = f" - {region_filter}"
    
    return html.Div([
        html.H4(f"Overview {title_suffix}{region_indicator}", style={'marginBottom': '20px', 'color': theme['text_primary']}),
        metrics_row,
        html.Hr(style={'borderColor': theme['border']}),
        html.H5(f"Country Comparison ({len(display_df)} countries)", 
                style={'marginTop': '20px', 'marginBottom': '15px', 'color': theme['text_primary']}),
        table
    ])

def create_timeseries_tab(selected_countries, region_filter, year_range, theme):
    """Time series tab with line charts"""
    # Apply country and region filters
    selected_countries = apply_country_region_filter(selected_countries, region_filter)
    
    # Check if we need to show info box and limit countries
    show_info_box = False
    original_count = len(selected_countries)
    
    if len(selected_countries) > 3:
        # Get top 3 countries by GDP from the filtered set
        top_countries_df = df[df['Country'].isin(selected_countries)].groupby('Country')['Gross Domestic Product (GDP)'].sum().nlargest(3)
        selected_countries = top_countries_df.index.tolist()
        show_info_box = True
    
    if not selected_countries:
        return html.Div("No countries match the selected filters", 
                       style={'textAlign': 'center', 'padding': '50px', 'color': theme['text_secondary']})
    
    filtered_df = df[
        (df['Country'].isin(selected_countries)) & 
        (df['Year'] >= year_range[0]) & 
        (df['Year'] <= year_range[1])
    ]
    
    # Info box for limited display
    info_box = html.Div()
    if show_info_box:
        region_text = f" in {region_filter}" if region_filter != 'all' else ""
        info_box = dbc.Alert([
            html.I(className="fas fa-info-circle", style={'marginRight': '10px'}),
            f"Showing top 3 of {original_count} countries{region_text} by GDP for readability: {', '.join(selected_countries)}. ",
            "Use Rankings or Map tabs for complete view."
        ], color="info", style={'marginBottom': '20px'})
    
    # GDP over time
    fig_gdp = px.line(
        filtered_df, 
        x='Year', 
        y='Gross Domestic Product (GDP)', 
        color='Country',
        title='GDP Evolution Over Time',
        template=theme['plotly_template']
    )
    fig_gdp.update_layout(
        yaxis_title='GDP (USD)',
        hovermode='x unified',
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary']
    )
    
    # Trade over time
    fig_trade = px.line(
        filtered_df, 
        x='Year', 
        y='Total_Trade', 
        color='Country',
        title='Total Trade Evolution Over Time',
        template=theme['plotly_template']
    )
    fig_trade.update_layout(
        yaxis_title='Total Trade (USD)',
        hovermode='x unified',
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary']
    )
    
    # Trade Openness over time
    fig_openness = px.line(
        filtered_df, 
        x='Year', 
        y='Trade_Openness', 
        color='Country',
        title='Trade Openness (% of GDP) Over Time',
        template=theme['plotly_template']
    )
    fig_openness.update_layout(
        yaxis_title='Trade Openness (%)',
        hovermode='x unified',
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary']
    )
    
    # GDP per Capita over time
    fig_gdppc = px.line(
        filtered_df, 
        x='Year', 
        y='GDP_per_Capita', 
        color='Country',
        title='GDP per Capita Over Time',
        template=theme['plotly_template']
    )
    fig_gdppc.update_layout(
        yaxis_title='GDP per Capita (USD)',
        hovermode='x unified',
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary']
    )
    
    return html.Div([
        html.H4("Time Series Analysis", style={'marginBottom': '20px', 'color': theme['text_primary']}),
        info_box,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_gdp), width=6),
            dbc.Col(dcc.Graph(figure=fig_trade), width=6),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_openness), width=6),
            dbc.Col(dcc.Graph(figure=fig_gdppc), width=6),
        ])
    ])

def create_correlation_tab(selected_countries, region_filter, year_range, theme):
    """Correlation tab with scatter plots"""
    # Apply country and region filters
    selected_countries = apply_country_region_filter(selected_countries, region_filter)
    
    # Check if we need to show info box and limit countries
    show_info_box = False
    original_count = len(selected_countries)
    
    if len(selected_countries) > 3:
        # Get top 3 countries by GDP from the filtered set
        top_countries_df = df[df['Country'].isin(selected_countries)].groupby('Country')['Gross Domestic Product (GDP)'].sum().nlargest(3)
        selected_countries = top_countries_df.index.tolist()
        show_info_box = True
    
    if not selected_countries:
        return html.Div("No countries match the selected filters", 
                       style={'textAlign': 'center', 'padding': '50px', 'color': theme['text_secondary']})
    
    filtered_df = df[
        (df['Country'].isin(selected_countries)) & 
        (df['Year'] >= year_range[0]) & 
        (df['Year'] <= year_range[1])
    ]
    
    # Info box for limited display
    info_box = html.Div()
    if show_info_box:
        region_text = f" in {region_filter}" if region_filter != 'all' else ""
        info_box = dbc.Alert([
            html.I(className="fas fa-info-circle", style={'marginRight': '10px'}),
            f"Showing top 3 of {original_count} countries{region_text} by GDP for readability: {', '.join(selected_countries)}. ",
            "Use Rankings or Map tabs for complete view."
        ], color="info", style={'marginBottom': '20px'})
    
    # GDP vs Trade scatter
    fig_scatter1 = px.scatter(
        filtered_df,
        x='Total_Trade',
        y='Gross Domestic Product (GDP)',
        color='Country',
        size='Population',
        hover_data=['Year', 'Region'],
        title='GDP vs Total Trade',
        template=theme['plotly_template'],
        trendline='ols'
    )
    fig_scatter1.update_layout(
        xaxis_title='Total Trade (USD)',
        yaxis_title='GDP (USD)',
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary']
    )
    
    # Trade Openness vs GDP per Capita
    fig_scatter2 = px.scatter(
        filtered_df,
        x='Trade_Openness',
        y='GDP_per_Capita',
        color='Country',
        hover_data=['Year', 'Region'],
        title='Trade Openness vs GDP per Capita',
        template=theme['plotly_template'],
        trendline='ols'
    )
    fig_scatter2.update_layout(
        xaxis_title='Trade Openness (%)',
        yaxis_title='GDP per Capita (USD)',
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary']
    )
    
    # Exports vs Imports
    fig_scatter3 = px.scatter(
        filtered_df,
        x='Imports of goods and services',
        y='Exports of goods and services',
        color='Country',
        hover_data=['Year'],
        title='Exports vs Imports',
        template=theme['plotly_template']
    )
    max_val = max(filtered_df['Imports of goods and services'].max(), 
                  filtered_df['Exports of goods and services'].max())
    fig_scatter3.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Balance Line',
        line=dict(dash='dash', color='red')
    ))
    fig_scatter3.update_layout(
        xaxis_title='Imports (USD)',
        yaxis_title='Exports (USD)',
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary']
    )
    
    return html.Div([
        html.H4("Correlation Analysis", style={'marginBottom': '20px', 'color': theme['text_primary']}),
        info_box,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_scatter1), width=12),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_scatter2), width=6),
            dbc.Col(dcc.Graph(figure=fig_scatter3), width=6),
        ])
    ])

def create_rankings_tab(year_filter, year_range, region_filter, theme):
    """Rankings tab with top AND bottom performers"""
    
    # Filter data based on year selection
    if year_filter == 'all':
        filtered_df = df[
            (df['Year'] >= year_range[0]) &
            (df['Year'] <= year_range[1])
        ]
        
        if region_filter != 'all':
            filtered_df = filtered_df[filtered_df['Region'] == region_filter]
        
        # Calculate TOTALS (sum) for GDP and Trade, averages for ratios
        ranking_data = filtered_df.groupby(['Country', 'Region']).agg({
            'Gross Domestic Product (GDP)': 'sum',
            'Total_Trade': 'sum',
            'Trade_Openness': 'mean',
            'GDP_per_Capita': 'mean'
        }).reset_index()
        
        title_suffix = f"Total {year_range[0]}-{year_range[1]}"
    else:
        ranking_data = df[df['Year'] == year_filter].copy()
        
        if region_filter != 'all':
            ranking_data = ranking_data[ranking_data['Region'] == region_filter]
        
        title_suffix = f"{year_filter}"
    
    # Add region indicator to title
    if region_filter != 'all':
        title_suffix += f" - {region_filter}"
    
    # Top 10 and Bottom 10 by GDP
    top_gdp = ranking_data.nlargest(10, 'Gross Domestic Product (GDP)')[
        ['Country', 'Gross Domestic Product (GDP)', 'Region']
    ].copy()
    top_gdp['GDP (Trillions)'] = top_gdp['Gross Domestic Product (GDP)'] / 1e12
    
    bottom_gdp = ranking_data.nsmallest(10, 'Gross Domestic Product (GDP)')[
        ['Country', 'Gross Domestic Product (GDP)', 'Region']
    ].copy()
    bottom_gdp['GDP (Billions)'] = bottom_gdp['Gross Domestic Product (GDP)'] / 1e9
    
    fig_top_gdp = px.bar(
        top_gdp.sort_values('GDP (Trillions)', ascending=True),
        x='GDP (Trillions)',
        y='Country',
        color='Region',
        title=f'Top 10 GDP ({title_suffix})',
        template=theme['plotly_template'],
        orientation='h'
    )
    fig_top_gdp.update_layout(
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary'],
        height=550
    )
    
    fig_bottom_gdp = px.bar(
        bottom_gdp.sort_values('GDP (Billions)', ascending=True),
        x='GDP (Billions)',
        y='Country',
        color='Region',
        title=f'Bottom 10 GDP ({title_suffix})',
        template=theme['plotly_template'],
        orientation='h'
    )
    fig_bottom_gdp.update_layout(
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary'],
        height=550
    )
    
    # Top 10 and Bottom 10 by Trade Volume
    top_trade = ranking_data.nlargest(10, 'Total_Trade')[
        ['Country', 'Total_Trade', 'Region']
    ].copy()
    top_trade['Trade (Billions)'] = top_trade['Total_Trade'] / 1e9
    
    bottom_trade = ranking_data.nsmallest(10, 'Total_Trade')[
        ['Country', 'Total_Trade', 'Region']
    ].copy()
    bottom_trade['Trade (Billions)'] = bottom_trade['Total_Trade'] / 1e9
    
    fig_top_trade = px.bar(
        top_trade.sort_values('Trade (Billions)', ascending=True),
        x='Trade (Billions)',
        y='Country',
        color='Region',
        title=f'Top 10 Trade Volume ({title_suffix})',
        template=theme['plotly_template'],
        orientation='h'
    )
    fig_top_trade.update_layout(
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary'],
        height=550
    )
    
    fig_bottom_trade = px.bar(
        bottom_trade.sort_values('Trade (Billions)', ascending=True),
        x='Trade (Billions)',
        y='Country',
        color='Region',
        title=f'Bottom 10 Trade Volume ({title_suffix})',
        template=theme['plotly_template'],
        orientation='h'
    )
    fig_bottom_trade.update_layout(
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary'],
        height=550
    )
    
    # Top 10 and Bottom 10 by Trade Openness
    top_openness = ranking_data.nlargest(10, 'Trade_Openness')[
        ['Country', 'Trade_Openness', 'Region']
    ]
    
    bottom_openness = ranking_data.nsmallest(10, 'Trade_Openness')[
        ['Country', 'Trade_Openness', 'Region']
    ]
    
    fig_top_openness = px.bar(
        top_openness.sort_values('Trade_Openness', ascending=True),
        x='Trade_Openness',
        y='Country',
        color='Region',
        title=f'Top 10 Trade Openness ({title_suffix})',
        template=theme['plotly_template'],
        orientation='h'
    )
    fig_top_openness.update_layout(
        xaxis_title='Trade Openness (%)',
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary'],
        height=550
    )
    
    fig_bottom_openness = px.bar(
        bottom_openness.sort_values('Trade_Openness', ascending=True),
        x='Trade_Openness',
        y='Country',
        color='Region',
        title=f'Bottom 10 Trade Openness ({title_suffix})',
        template=theme['plotly_template'],
        orientation='h'
    )
    fig_bottom_openness.update_layout(
        xaxis_title='Trade Openness (%)',
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary'],
        height=550
    )
    
    # Top 10 and Bottom 10 by GDP per Capita
    top_gdppc = ranking_data.nlargest(10, 'GDP_per_Capita')[
        ['Country', 'GDP_per_Capita', 'Region']
    ]
    
    bottom_gdppc = ranking_data.nsmallest(10, 'GDP_per_Capita')[
        ['Country', 'GDP_per_Capita', 'Region']
    ]
    
    fig_top_gdppc = px.bar(
        top_gdppc.sort_values('GDP_per_Capita', ascending=True),
        x='GDP_per_Capita',
        y='Country',
        color='Region',
        title=f'Top 10 GDP per Capita ({title_suffix})',
        template=theme['plotly_template'],
        orientation='h'
    )
    fig_top_gdppc.update_layout(
        xaxis_title='GDP per Capita (USD)',
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary'],
        height=550
    )
    
    fig_bottom_gdppc = px.bar(
        bottom_gdppc.sort_values('GDP_per_Capita', ascending=True),
        x='GDP_per_Capita',
        y='Country',
        color='Region',
        title=f'Bottom 10 GDP per Capita ({title_suffix})',
        template=theme['plotly_template'],
        orientation='h'
    )
    fig_bottom_gdppc.update_layout(
        xaxis_title='GDP per Capita (USD)',
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary'],
        height=550
    )
    
    return html.Div([
        html.H4(f"Global Rankings ({title_suffix})", style={'marginBottom': '20px', 'color': theme['text_primary']}),
        
        html.H5("📊 GDP Rankings", style={'marginTop': '20px', 'marginBottom': '15px', 'color': theme['text_primary']}),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_top_gdp), width=6),
            dbc.Col(dcc.Graph(figure=fig_bottom_gdp), width=6),
        ]),
        
        html.H5("🌐 Trade Volume Rankings", style={'marginTop': '30px', 'marginBottom': '15px', 'color': theme['text_primary']}),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_top_trade), width=6),
            dbc.Col(dcc.Graph(figure=fig_bottom_trade), width=6),
        ]),
        
        html.H5("📈 Trade Openness Rankings", style={'marginTop': '30px', 'marginBottom': '15px', 'color': theme['text_primary']}),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_top_openness), width=6),
            dbc.Col(dcc.Graph(figure=fig_bottom_openness), width=6),
        ]),
        
        html.H5("💰 GDP per Capita Rankings", style={'marginTop': '30px', 'marginBottom': '15px', 'color': theme['text_primary']}),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_top_gdppc), width=6),
            dbc.Col(dcc.Graph(figure=fig_bottom_gdppc), width=6),
        ])
    ])

def create_map_tab(year_filter, year_range, region_filter, theme):
    """Map tab with choropleth"""
    
    if year_filter == 'all':
        map_data = df[
            (df['Year'] >= year_range[0]) &
            (df['Year'] <= year_range[1])
        ].groupby('Country').agg({
            'Trade_Openness': 'mean',
            'GDP_per_Capita': 'mean',
            'Gross Domestic Product (GDP)': 'mean',
            'Region': 'first'
        }).reset_index()
        
        title_suffix = f"(Average {year_range[0]}-{year_range[1]})"
    else:
        map_data = df[df['Year'] == year_filter].copy()
        title_suffix = f"({year_filter})"
    
    # Apply region filter if selected
    if region_filter != 'all':
        map_data = map_data[map_data['Region'] == region_filter]
        title_suffix += f" - {region_filter}"
    
    # Trade Openness Map
    fig_map1 = px.choropleth(
        map_data,
        locations='Country',
        locationmode='country names',
        color='Trade_Openness',
        hover_name='Country',
        hover_data={
            'Trade_Openness': ':.1f',
            'Gross Domestic Product (GDP)': ':,.0f',
            'Region': True
        },
        title=f'Global Trade Openness Map {title_suffix}',
        color_continuous_scale='RdYlGn',
        template=theme['plotly_template']
    )
    fig_map1.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'),
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary']
    )
    
    # GDP per Capita Map
    fig_map2 = px.choropleth(
        map_data,
        locations='Country',
        locationmode='country names',
        color='GDP_per_Capita',
        hover_name='Country',
        hover_data={
            'GDP_per_Capita': ':,.0f',
            'Gross Domestic Product (GDP)': ':,.0f',
            'Region': True
        },
        title=f'Global GDP per Capita Map {title_suffix}',
        color_continuous_scale='Blues',
        template=theme['plotly_template']
    )
    fig_map2.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'),
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary']
    )
    
    return html.Div([
        html.H4(f"Global Maps {title_suffix}", style={'marginBottom': '20px', 'color': theme['text_primary']}),
        dcc.Graph(figure=fig_map1),
        dcc.Graph(figure=fig_map2)
    ])

def create_clusters_tab(theme):
    """Clusters tab showing country groupings with member lists"""
    if not has_clusters:
        return html.Div("Cluster data not available", 
                       style={'textAlign': 'center', 'padding': '50px', 'color': theme['text_secondary']})
    
    # Cluster distribution
    cluster_counts = df_cluster['Cluster_Name'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    
    fig_cluster_dist = px.bar(
        cluster_counts,
        x='Cluster',
        y='Count',
        title='Country Distribution by Cluster',
        template=theme['plotly_template'],
        color='Cluster'
    )
    fig_cluster_dist.update_layout(
        xaxis_title='Cluster Type',
        yaxis_title='Number of Countries',
        showlegend=False,
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary']
    )
    
    # Cluster characteristics
    fig_cluster_chars = px.scatter(
        df_cluster,
        x='Trade_Openness',
        y='GDP_per_Capita',
        color='Cluster_Name',
        size='Gross Domestic Product (GDP)',
        hover_data=['Country', 'Region'],
        title='Cluster Characteristics: Trade Openness vs GDP per Capita',
        template=theme['plotly_template']
    )
    fig_cluster_chars.update_layout(
        xaxis_title='Trade Openness (%)',
        yaxis_title='GDP per Capita (USD)',
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary']
    )
    
    # Regional distribution by cluster
    cluster_region = pd.crosstab(df_cluster['Cluster_Name'], df_cluster['Region'])
    
    fig_cluster_region = px.bar(
        cluster_region,
        title='Regional Distribution Across Clusters',
        template=theme['plotly_template'],
        barmode='stack'
    )
    fig_cluster_region.update_layout(
        xaxis_title='Cluster',
        yaxis_title='Number of Countries',
        plot_bgcolor=theme['bg_primary'],
        paper_bgcolor=theme['bg_primary'],
        font_color=theme['text_primary']
    )
    
    # Create country lists for each cluster
    cluster_details = []
    for cluster_name in sorted(df_cluster['Cluster_Name'].unique()):
        cluster_countries = df_cluster[df_cluster['Cluster_Name'] == cluster_name].sort_values('Country')
        country_list = cluster_countries['Country'].tolist()
        
        cluster_card = dbc.Card([
            dbc.CardHeader(
                html.H5(f"{cluster_name} ({len(country_list)} countries)", 
                       style={'marginBottom': '0', 'color': theme['text_primary']})
            ),
            dbc.CardBody([
                html.P(f"Member Countries:", style={'fontWeight': 'bold', 'color': theme['text_primary']}),
                html.Div([
                    html.Span(f"{country}, ", style={'color': theme['text_secondary']})
                    for country in country_list[:-1]
                ] + [html.Span(country_list[-1], style={'color': theme['text_secondary']})] if len(country_list) > 0 else [])
            ])
        ], style={
            'marginBottom': '15px',
            'backgroundColor': theme['bg_secondary'],
            'border': f"1px solid {theme['border']}"
        })
        
        cluster_details.append(cluster_card)
    
    return html.Div([
        html.H4("Cluster Analysis", style={'marginBottom': '20px', 'color': theme['text_primary']}),
        html.P("Countries grouped by similar trade and economic patterns", 
               style={'color': theme['text_secondary'], 'marginBottom': '20px'}),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_cluster_dist), width=12),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_cluster_chars), width=12),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_cluster_region), width=12),
        ]),
        
        html.Hr(style={'borderColor': theme['border'], 'marginTop': '30px'}),
        
        html.H5("📋 Cluster Members", style={'marginTop': '30px', 'marginBottom': '20px', 'color': theme['text_primary']}),
        html.Div(cluster_details)
    ])

def create_glossary_tab(theme):
    """Glossary tab with term definitions"""
    
    terms = [
        {
            'term': 'GDP (Gross Domestic Product)',
            'definition': 'The total monetary value of all goods and services produced within a country\'s borders in a specific time period. It measures the size of an economy.',
            'formula': 'GDP = Consumption + Investment + Government Spending + (Exports - Imports)'
        },
        {
            'term': 'Trade Openness',
            'definition': 'The ratio of total trade (exports + imports) to GDP, expressed as a percentage. It measures how integrated a country is in global trade.',
            'formula': 'Trade Openness = (Total Trade / GDP) × 100'
        },
        {
            'term': 'GDP per Capita',
            'definition': 'The average economic output per person, calculated by dividing GDP by population. It indicates the average wealth or living standards in a country.',
            'formula': 'GDP per Capita = GDP / Population'
        },
        {
            'term': 'Total Trade',
            'definition': 'The sum of a country\'s exports and imports of goods and services. It represents the total volume of international trade.',
            'formula': 'Total Trade = Exports + Imports'
        },
        {
            'term': 'Trade Balance',
            'definition': 'The difference between a country\'s exports and imports. A positive balance (surplus) means exports exceed imports; negative (deficit) means imports exceed exports.',
            'formula': 'Trade Balance = Exports - Imports'
        },
        {
            'term': 'Net Exporter',
            'definition': 'A country that exports more than it imports, resulting in a trade surplus. These countries have a positive trade balance.',
            'formula': 'Net Exporter: Exports > Imports'
        },
        {
            'term': 'Net Importer',
            'definition': 'A country that imports more than it exports, resulting in a trade deficit. These countries have a negative trade balance.',
            'formula': 'Net Importer: Imports > Exports'
        },
        {
            'term': 'Export/Import Ratio',
            'definition': 'The ratio of exports to imports. A ratio greater than 1 indicates a trade surplus; less than 1 indicates a deficit.',
            'formula': 'Export/Import Ratio = Exports / Imports'
        },
        {
            'term': 'Trade per Capita',
            'definition': 'The average trade volume per person, showing how much trade occurs relative to population size.',
            'formula': 'Trade per Capita = Total Trade / Population'
        },
        {
            'term': 'Observations',
            'definition': 'Country-year combinations in the dataset. For example, "USA in 2020" is one observation. With 220 countries and 52 years, we have 10,512 observations.',
            'formula': 'Observations = Countries × Years'
        },
        {
            'term': 'Correlation',
            'definition': 'A statistical measure of the relationship between two variables, ranging from -1 to +1. Values near +1 indicate strong positive correlation.',
            'formula': 'Pearson r = Covariance(X,Y) / (StdDev(X) × StdDev(Y))'
        },
        {
            'term': 'Granger Causality',
            'definition': 'A statistical test to determine if past values of one variable help predict future values of another. It indicates predictive relationships, not true causation.',
            'formula': 'Tests if X(t-1) predicts Y(t) beyond what Y(t-1) already predicts'
        }
    ]
    
    term_cards = []
    for term_data in terms:
        card = dbc.Card([
            dbc.CardHeader(
                html.H5(term_data['term'], style={'color': theme['accent'], 'marginBottom': '0'})
            ),
            dbc.CardBody([
                html.P(term_data['definition'], style={'color': theme['text_primary'], 'marginBottom': '10px'}),
                html.Div([
                    html.Strong("Formula: ", style={'color': theme['text_secondary']}),
                    html.Code(term_data['formula'], style={
                        'backgroundColor': theme['bg_primary'],
                        'padding': '5px 10px',
                        'borderRadius': '5px',
                        'color': theme['text_primary']
                    })
                ])
            ])
        ], style={
            'marginBottom': '15px',
            'backgroundColor': theme['bg_secondary'],
            'border': f"1px solid {theme['border']}"
        })
        term_cards.append(card)
    
    return html.Div([
        html.H4("📚 Glossary - Key Terms & Definitions", style={'marginBottom': '20px', 'color': theme['text_primary']}),
        html.P("Understanding the metrics and terminology used in this dashboard", 
               style={'color': theme['text_secondary'], 'marginBottom': '30px', 'fontSize': '1.1rem'}),
        html.Div(term_cards)
    ])

# ==============================================================================
# APP LAYOUT
# ==============================================================================

initial_theme = LIGHT_THEME

app.layout = html.Div([
    dcc.Store(id='theme-store', data='light'),
    dcc.Download(id="download-pdf"),
    html.Div([
        create_navbar(initial_theme),
        dbc.Container([
            dbc.Row([
                dbc.Col(create_sidebar(initial_theme), width=3),
                dbc.Col(create_main_content(initial_theme), width=9)
            ])
        ], fluid=True, style={'backgroundColor': initial_theme['bg_primary'], 'minHeight': '100vh'}),
        
        html.Footer([
            html.Hr(style={'borderColor': initial_theme['border'], 'margin': '0'}),
            html.Div([
                html.P([
                    "Designed by ",
                    html.A("Osasere Edobor", 
                           href="https://portfolio.edoborosasere.com",
                           target="_blank",
                           style={
                               'color': initial_theme['accent'],
                               'textDecoration': 'none',
                               'fontWeight': 'bold'
                           }),
                    " | © 2024"
                ], style={
                    'textAlign': 'center',
                    'padding': '20px',
                    'color': initial_theme['text_secondary'],
                    'margin': '0'
                })
            ], style={'backgroundColor': initial_theme['bg_secondary']})
        ])
    ], id='app-container')
])

# ==============================================================================
# CALLBACKS
# ==============================================================================

@app.callback(
    Output('app-container', 'children'),
    Input('theme-store', 'data'),
    prevent_initial_call=True
)
def update_layout(theme_mode):
    """Update entire layout based on theme"""
    theme = DARK_THEME if theme_mode == 'dark' else LIGHT_THEME
    
    return html.Div([
        create_navbar(theme),
        dbc.Container([
            dbc.Row([
                dbc.Col(create_sidebar(theme), width=3),
                dbc.Col(create_main_content(theme), width=9)
            ])
        ], fluid=True, style={'backgroundColor': theme['bg_primary'], 'minHeight': '100vh'}),
        
        html.Footer([
            html.Hr(style={'borderColor': theme['border'], 'margin': '0'}),
            html.Div([
                html.P([
                    "Designed by ",
                    html.A("Osasere Edobor", 
                           href="https://portfolio.edoborosasere.com",
                           target="_blank",
                           style={
                               'color': theme['accent'],
                               'textDecoration': 'none',
                               'fontWeight': 'bold'
                           }),
                    " | © 2024"
                ], style={
                    'textAlign': 'center',
                    'padding': '20px',
                    'color': theme['text_secondary'],
                    'margin': '0'
                })
            ], style={'backgroundColor': theme['bg_secondary']})
        ])
    ])

@app.callback(
    [Output('theme-store', 'data'),
     Output('theme-icon', 'className')],
    Input('theme-toggle', 'n_clicks'),
    State('theme-store', 'data'),
    prevent_initial_call=True
)
def toggle_theme(n_clicks, current_theme):
    """Toggle between light and dark theme"""
    if current_theme == 'light':
        return 'dark', 'fas fa-sun'
    else:
        return 'light', 'fas fa-moon'

@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'active_tab'),
     Input('country-dropdown', 'value'),
     Input('region-dropdown', 'value'),
     Input('year-dropdown', 'value'),
     Input('year-slider', 'value'),
     Input('theme-store', 'data')]
)
def render_tab_content(active_tab, selected_countries, region_filter, year_filter, year_range, theme_mode):
    """Render content based on active tab"""
    theme = DARK_THEME if theme_mode == 'dark' else LIGHT_THEME
    
    if active_tab == 'overview':
        return create_overview_tab(selected_countries, region_filter, year_filter, year_range, theme)
    elif active_tab == 'timeseries':
        return create_timeseries_tab(selected_countries, region_filter, year_range, theme)
    elif active_tab == 'correlation':
        return create_correlation_tab(selected_countries, region_filter, year_range, theme)
    elif active_tab == 'rankings':
        return create_rankings_tab(year_filter, year_range, region_filter, theme)
    elif active_tab == 'map':
        return create_map_tab(year_filter, year_range, region_filter, theme)
    elif active_tab == 'glossary':
        return create_glossary_tab(theme)
    elif active_tab == 'clusters':
        return create_clusters_tab(theme)
    
    return html.Div("Select a tab", style={'color': theme['text_secondary']})

# ==============================================================================
# PDF EXPORT CALLBACK - COMPREHENSIVE REPORT
# ==============================================================================

@app.callback(
    Output("download-pdf", "data"),
    Input("export-pdf-btn", "n_clicks"),
    [State('country-dropdown', 'value'),
     State('region-dropdown', 'value'),
     State('year-dropdown', 'value'),
     State('year-slider', 'value')],
    prevent_initial_call=True
)
def export_comprehensive_report(n_clicks, selected_countries, region_filter, year_filter, year_range):
    """Generate comprehensive PDF report with all analysis sections"""
    
    # Apply country and region filters
    selected_countries = apply_country_region_filter(selected_countries, region_filter)
    
    if not selected_countries:
        return None
    
    # Create PDF in memory
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []
    
    # ==================== STYLES ====================
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#0D6EFD'),
        spaceAfter=20,
        alignment=1,  # Center
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#212529'),
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#495057'),
        spaceAfter=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#212529')
    )
    
    # ==================== COVER PAGE ====================
    
    # Title
    title = Paragraph("Trade & GDP Analysis Report", title_style)
    story.append(title)
    story.append(Spacer(1, 0.3*inch))
    
    # Report metadata
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    if year_filter == 'all':
        time_period = f"{year_range[0]}-{year_range[1]}"
    else:
        time_period = str(year_filter)
    
    # Warning for large reports
    warning_text = ""
    if len(selected_countries) > 20:
        warning_text = f"<br/><b style='color: red;'>⚠️  Large Report: This report includes {len(selected_countries)} countries and may be extensive.</b><br/>"
    elif len(selected_countries) > 10:
        warning_text = f"<br/><b style='color: orange;'>Note: This report includes {len(selected_countries)} countries.</b><br/>"
    
    region_text = f"<b>Region Filter:</b> {region_filter}<br/>" if region_filter != 'all' else ""
    
    metadata_text = f"""
    <b>Generated:</b> {report_date}<br/>
    <b>Selected Countries:</b> {', '.join(selected_countries[:10])}{'...' if len(selected_countries) > 10 else ''}<br/>
    {region_text}
    <b>Time Period:</b> {time_period}<br/>
    <b>Total Countries:</b> {len(selected_countries)} {'country' if len(selected_countries) == 1 else 'countries'}<br/>
    {warning_text}
    """
    story.append(Paragraph(metadata_text, body_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Filter data
    if year_filter == 'all':
        filtered_df = df[
            (df['Country'].isin(selected_countries)) &
            (df['Year'] >= year_range[0]) &
            (df['Year'] <= year_range[1])
        ]
    else:
        filtered_df = df[
            (df['Country'].isin(selected_countries)) &
            (df['Year'] == year_filter)
        ]
    
    # ==================== EXECUTIVE SUMMARY ====================
    
    story.append(Paragraph("Executive Summary", heading_style))
    
    total_gdp = filtered_df['Gross Domestic Product (GDP)'].sum()
    total_trade = filtered_df['Total_Trade'].sum()
    avg_openness = filtered_df['Trade_Openness'].mean()
    avg_gdp_pc = filtered_df['GDP_per_Capita'].mean()
    
    region_context = f" in {region_filter}" if region_filter != 'all' else ""
    
    summary_text = f"""
    This report analyzes the trade and economic performance of {len(selected_countries)} 
    {'country' if len(selected_countries) == 1 else 'countries'}{region_context} over the period {time_period}. 
    Key findings include a combined GDP of <b>{format_number(total_gdp, prefix='$')}</b>, 
    total trade volume of <b>{format_number(total_trade, prefix='$')}</b>, 
    average trade openness of <b>{avg_openness:.1f}%</b>, and average GDP per capita of 
    <b>{format_number(avg_gdp_pc, prefix='$', decimals=0)}</b>.
    """
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Summary metrics table
    summary_data = [
        ['Metric', 'Value'],
        ['Total GDP', format_number(total_gdp, prefix='$', decimals=2)],
        ['Total Trade', format_number(total_trade, prefix='$', decimals=2)],
        ['Avg Trade Openness', f"{avg_openness:.1f}%"],
        ['Avg GDP per Capita', format_number(avg_gdp_pc, prefix='$', decimals=0)]
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0D6EFD')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white])
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 0.4*inch))
    
    # ==================== COUNTRY COMPARISON ====================
    
    story.append(Paragraph("Country Comparison", heading_style))
    
    if year_filter == 'all':
        comparison_df = filtered_df.groupby('Country').agg({
            'Gross Domestic Product (GDP)': 'sum',
            'Total_Trade': 'sum',
            'Trade_Openness': 'mean',
            'GDP_per_Capita': 'mean'
        }).reset_index().sort_values('Gross Domestic Product (GDP)', ascending=False)
    else:
        comparison_df = filtered_df[[
            'Country', 'Gross Domestic Product (GDP)', 'Total_Trade', 
            'Trade_Openness', 'GDP_per_Capita'
        ]].copy().sort_values('Gross Domestic Product (GDP)', ascending=False)
    
    # Limit to top 20 for PDF readability if more than 20 countries
    if len(comparison_df) > 20:
        story.append(Paragraph(f"<i>Showing top 20 of {len(comparison_df)} countries by GDP</i>", body_style))
        story.append(Spacer(1, 0.1*inch))
        comparison_df = comparison_df.head(20)
    
    # Format country comparison table
    table_data = [['Country', 'GDP', 'Trade', 'Openness', 'GDP/Capita']]
    for _, row in comparison_df.iterrows():
        table_data.append([
            row['Country'][:20],  # Truncate long names
            f"${row['Gross Domestic Product (GDP)']/1e9:,.1f}B",
            f"${row['Total_Trade']/1e9:,.1f}B",
            f"{row['Trade_Openness']:.1f}%",
            f"${row['GDP_per_Capita']:,.0f}"
        ])
    
    country_table = Table(table_data, colWidths=[1.8*inch, 1.4*inch, 1.4*inch, 1*inch, 1.4*inch])
    country_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#198754')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightgrey, colors.white])
    ]))
    
    story.append(country_table)
    story.append(Spacer(1, 0.4*inch))
    
    # ==================== TIME SERIES INSIGHTS ====================
    
    if year_filter == 'all' and len(filtered_df) > 1:
        story.append(Paragraph("Time Series Analysis", heading_style))
        
        # Limit to top 5 countries for PDF
        top_countries_for_ts = filtered_df.groupby('Country')['Gross Domestic Product (GDP)'].sum().nlargest(5).index.tolist()
        
        if len(selected_countries) > 5:
            story.append(Paragraph(f"<i>Showing top 5 of {len(selected_countries)} countries by GDP</i>", body_style))
            story.append(Spacer(1, 0.1*inch))
        
        # Calculate growth rates
        for country in top_countries_for_ts:
            country_data = filtered_df[filtered_df['Country'] == country].sort_values('Year')
            if len(country_data) >= 2:
                first_year = country_data.iloc[0]
                last_year = country_data.iloc[-1]
                
                gdp_growth = ((last_year['Gross Domestic Product (GDP)'] / first_year['Gross Domestic Product (GDP)']) - 1) * 100
                trade_growth = ((last_year['Total_Trade'] / first_year['Total_Trade']) - 1) * 100
                
                insight_text = f"""
                <b>{country}:</b><br/>
                • GDP grew by <b>{gdp_growth:.1f}%</b> from {format_number(first_year['Gross Domestic Product (GDP)'], prefix='$')} 
                  to {format_number(last_year['Gross Domestic Product (GDP)'], prefix='$')}<br/>
                • Trade volume increased by <b>{trade_growth:.1f}%</b> from {format_number(first_year['Total_Trade'], prefix='$')} 
                  to {format_number(last_year['Total_Trade'], prefix='$')}<br/>
                • Trade openness changed from <b>{first_year['Trade_Openness']:.1f}%</b> to <b>{last_year['Trade_Openness']:.1f}%</b>
                """
                story.append(Paragraph(insight_text, body_style))
                story.append(Spacer(1, 0.15*inch))
        
        story.append(Spacer(1, 0.2*inch))
    
    # ==================== CORRELATION INSIGHTS ====================
    
    story.append(Paragraph("Correlation Analysis", heading_style))
    
    # Calculate correlations
    gdp_trade_corr = filtered_df['Gross Domestic Product (GDP)'].corr(filtered_df['Total_Trade'])
    gdp_exports_corr = filtered_df['Gross Domestic Product (GDP)'].corr(filtered_df['Exports of goods and services'])
    gdp_imports_corr = filtered_df['Gross Domestic Product (GDP)'].corr(filtered_df['Imports of goods and services'])
    openness_gdppc_corr = filtered_df['Trade_Openness'].corr(filtered_df['GDP_per_Capita'])
    
    corr_text = f"""
    Statistical analysis reveals the following correlations:<br/><br/>
    
    <b>GDP vs Total Trade:</b> r = {gdp_trade_corr:.3f}<br/>
    {'Strong positive correlation - higher trade volumes are associated with larger economies.' if abs(gdp_trade_corr) > 0.7 else 'Moderate positive correlation between trade and economic size.'}<br/><br/>
    
    <b>GDP vs Exports:</b> r = {gdp_exports_corr:.3f}<br/>
    <b>GDP vs Imports:</b> r = {gdp_imports_corr:.3f}<br/>
    {'Imports show slightly stronger correlation than exports.' if gdp_imports_corr > gdp_exports_corr else 'Exports show slightly stronger correlation than imports.'}<br/><br/>
    
    <b>Trade Openness vs GDP per Capita:</b> r = {openness_gdppc_corr:.3f}<br/>
    {'Positive correlation suggests more open economies tend to be wealthier per capita.' if openness_gdppc_corr > 0.3 else 'Weak or negative correlation between openness and wealth.'}
    """
    
    story.append(Paragraph(corr_text, body_style))
    story.append(Spacer(1, 0.4*inch))
    
    # ==================== RANKINGS ====================
    
    story.append(Paragraph("Global Rankings", heading_style))
    
    # Prepare ranking data
    if year_filter == 'all':
        ranking_df = filtered_df.groupby('Country').agg({
            'Gross Domestic Product (GDP)': 'sum',
            'Total_Trade': 'sum',
            'Trade_Openness': 'mean',
            'GDP_per_Capita': 'mean'
        }).reset_index()
    else:
        ranking_df = filtered_df.copy()
    
    # TOP GDP
    story.append(Paragraph("Top Countries by GDP", subheading_style))
    top_gdp = ranking_df.nlargest(min(5, len(ranking_df)), 'Gross Domestic Product (GDP)')
    
    gdp_data = [['Rank', 'Country', 'GDP']]
    for i, (_, row) in enumerate(top_gdp.iterrows(), 1):
        gdp_data.append([
            str(i),
            row['Country'][:25],
            f"${row['Gross Domestic Product (GDP)']/1e9:,.1f}B"
        ])
    
    gdp_table = Table(gdp_data, colWidths=[0.8*inch, 3*inch, 2.2*inch])
    gdp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0D6EFD')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightblue, colors.white])
    ]))
    story.append(gdp_table)
    story.append(Spacer(1, 0.25*inch))
    
    # TOP TRADE
    story.append(Paragraph("Top Countries by Trade Volume", subheading_style))
    top_trade = ranking_df.nlargest(min(5, len(ranking_df)), 'Total_Trade')
    
    trade_data = [['Rank', 'Country', 'Trade Volume']]
    for i, (_, row) in enumerate(top_trade.iterrows(), 1):
        trade_data.append([
            str(i),
            row['Country'][:25],
            f"${row['Total_Trade']/1e9:,.1f}B"
        ])
    
    trade_table = Table(trade_data, colWidths=[0.8*inch, 3*inch, 2.2*inch])
    trade_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#198754')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightgreen, colors.white])
    ]))
    story.append(trade_table)
    story.append(Spacer(1, 0.25*inch))
    
    # TOP OPENNESS
    story.append(Paragraph("Most Trade-Open Economies", subheading_style))
    top_openness = ranking_df.nlargest(min(5, len(ranking_df)), 'Trade_Openness')
    
    openness_data = [['Rank', 'Country', 'Openness']]
    for i, (_, row) in enumerate(top_openness.iterrows(), 1):
        openness_data.append([
            str(i),
            row['Country'][:25],
            f"{row['Trade_Openness']:.1f}%"
        ])
    
    openness_table = Table(openness_data, colWidths=[0.8*inch, 3*inch, 2.2*inch])
    openness_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FFC107')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightyellow, colors.white])
    ]))
    story.append(openness_table)
    story.append(Spacer(1, 0.25*inch))
    
    # TOP GDP PER CAPITA
    story.append(Paragraph("Wealthiest Countries (GDP per Capita)", subheading_style))
    top_gdppc = ranking_df.nlargest(min(5, len(ranking_df)), 'GDP_per_Capita')
    
    gdppc_data = [['Rank', 'Country', 'GDP per Capita']]
    for i, (_, row) in enumerate(top_gdppc.iterrows(), 1):
        gdppc_data.append([
            str(i),
            row['Country'][:25],
            f"${row['GDP_per_Capita']:,.0f}"
        ])
    
    gdppc_table = Table(gdppc_data, colWidths=[0.8*inch, 3*inch, 2.2*inch])
    gdppc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6F42C1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lavender, colors.white])
    ]))
    story.append(gdppc_table)
    story.append(Spacer(1, 0.4*inch))
    
    # ==================== CLUSTER ANALYSIS ====================
    
    if has_clusters:
        # Check if any selected countries are in cluster data
        cluster_countries = df_cluster[df_cluster['Country'].isin(selected_countries)]
        
        if len(cluster_countries) > 0:
            story.append(Paragraph("Cluster Analysis", heading_style))
            
            cluster_text = f"""
            Based on trade patterns and economic characteristics, the selected countries 
            fall into the following clusters:
            """
            story.append(Paragraph(cluster_text, body_style))
            story.append(Spacer(1, 0.15*inch))
            
            # Limit to top 15 for PDF
            if len(cluster_countries) > 15:
                story.append(Paragraph(f"<i>Showing 15 of {len(cluster_countries)} countries</i>", body_style))
                story.append(Spacer(1, 0.1*inch))
                cluster_countries = cluster_countries.head(15)
            
            # Show cluster assignments
            cluster_data = [['Country', 'Cluster Type', 'Trade Openness', 'GDP/Capita']]
            for _, row in cluster_countries.iterrows():
                cluster_data.append([
                    row['Country'][:20],
                    row['Cluster_Name'][:30],
                    f"{row['Trade_Openness']:.1f}%",
                    f"${row['GDP_per_Capita']:,.0f}"
                ])
            
            cluster_table = Table(cluster_data, colWidths=[1.5*inch, 2.5*inch, 1.2*inch, 1.3*inch])
            cluster_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#17A2B8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightblue, colors.white])
            ]))
            story.append(cluster_table)
            story.append(Spacer(1, 0.3*inch))
    
    # ==================== FOOTER ====================
    
    story.append(Spacer(1, 0.5*inch))
    
    footer_text = """
    <b>Methodology Note:</b> This report is generated from the Trade & GDP Analysis Dashboard, 
    which analyzes data from 220 countries spanning 1970-2021. Correlations are calculated using 
    Pearson correlation coefficient. All monetary values are in USD.<br/><br/>
    
    <i>Generated by Trade & GDP Analysis Dashboard | Designed by Osasere Edobor<br/>
    Portfolio: portfolio.edoborosasere.com | Email: projects@edoborosasere.com</i>
    """
    
    footer = Paragraph(footer_text, body_style)
    story.append(footer)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    # Generate filename
    if len(selected_countries) <= 3:
        countries_str = "_".join([c.replace(" ", "").replace(",", "") for c in selected_countries])
    else:
        countries_str = f"{len(selected_countries)}_countries"
    
    if region_filter != 'all':
        countries_str += f"_{region_filter.replace(' ', '_')}"
    
    filename = f"Trade_GDP_Report_{countries_str}_{time_period}.pdf"
    
    return dcc.send_bytes(buffer.getvalue(), filename)

# ==============================================================================
# RUN APP
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Starting Dashboard...")
    print("="*80)
    print(f"\n📊 Dashboard will open at: http://127.0.0.1:8050/")
    print(f"   Press CTRL+C to stop the server\n")
    print("="*80 + "\n")
    
    app.run(debug=True, port=8050)