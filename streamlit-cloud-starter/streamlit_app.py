import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json, re
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import altair as alt # Import altair for interactive charts

st.set_page_config(page_title="Wealth Advisor Dashboard", layout="wide")

# Add custom CSS for better font clarity
st.markdown("""
<style>
    /* Improve font rendering and clarity */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 16px;
        font-weight: 400;
        line-height: 1.4;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
    }
    
    /* Improve table readability */
    .dataframe {
        font-size: 14px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .dataframe th {
        font-weight: 600;
        background-color: #f0f2f6;
    }
    
    .dataframe td {
        font-size: 14px;
        line-height: 1.3;
    }
    
    /* Improve headers */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        color: #262730;
    }
    
    /* Improve sidebar */
    .css-1d391kg, .css-1lcbmhc {
        font-size: 14px;
    }
    
    /* Improve select boxes and inputs */
    .stSelectbox, .stTextInput, .stNumberInput {
        font-size: 14px;
    }
    
    /* Better contrast for text */
    .stMarkdown, .stText, .stAlert {
        color: #262730;
    }
    
    /* Improve dataframes */
    .stDataFrame {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

PROCESSED_SUMMARIES_PATH = Path('/content/drive/MyDrive/Radiant/Output/processed_summaries.csv')


# ────────────────────────── Helpers ──────────────────────────
@st.cache_data
def load_json(p: Path):
    if not p.exists():
        st.error(f'❌ File not found: {p}'); st.stop()
    with p.open(encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def summary_stats(p: Path):
    cnt, earliest, latest, date_rx = 0, None, None, re.compile(r'\d{4}-\d{2}-\d{2}')
    if not p.exists(): return cnt, None, None
    for raw in p.read_text(encoding='utf-8').splitlines():
        raw = raw.strip()
        if not raw: continue
        cnt += 1
        try:
            obj = json.loads(raw)
            dstr = next((obj[k] for k in obj if k.lower().startswith('publish') and 'date' in k.lower()), None)
        except json.JSONDecodeError:
            m = date_rx.search(raw); dstr = m.group(0) if m else None
        if not dstr: continue
        try: d = datetime.fromisoformat(str(dstr)[:10])
        except ValueError: continue
        earliest = d if not earliest or d < earliest else earliest
        latest    = d if not latest   or d > latest   else latest
    return cnt, earliest, latest

def boxed_names(lst):
    if not lst:
        return '<i>None</i>'
    return ' '.join(
        "<span style='display:inline-block;padding:6px 12px;margin:4px;"
        "border:1px solid #e0e0e0;border-radius:6px;background:#fafafa;'>{}</span>".format(x)
        for x in lst
    )

# Create a function to format the DataFrame for better display with HTML/CSS
def format_dataframe_for_display(df, columns):
    # Apply CSS styling to prevent text wrapping and adjust column widths
    # Modified to prevent text wrapping in the 'Summary' column
    return df[columns].style.set_properties(**{
        'white-space': 'nowrap',
        'font-family': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif',
        'font-size': '14px'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('min-width', '150px'),
            ('font-weight', '600'),
            ('background-color', '#f0f2f6'),
            ('font-family', 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif')
        ]},
        {'selector': 'td', 'props': [
            ('min-width', '150px'),
            ('font-family', 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'),
            ('font-size', '14px')
        ]}
    ]).set_properties(subset=['Summary'], **{
        'min-width': '700px', 
        'white-space': 'normal', 
        'word-break': 'break-word',
        'font-family': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif',
        'font-size': '14px',
        'line-height': '1.4'
    })

# Modified to format month and year
def format_month_year(month_year_str):
    try:
        # Assuming month_year_str is in 'MM_YYYY' format or similar that can be parsed
        month, year = map(int, month_year_str.split('_'))
        return datetime(year, month, 1).strftime('%B %Y') # Format as "Month Year"
    except (ValueError, IndexError):
        return month_year_str # Return original if format is unexpected

# ────────────────────────── Load data ─────────────────────────
@st.cache_data
def load_summaries(p: Path):
    if not p.exists():
        st.error(f'❌ File not found: {p}'); st.stop()
    # Correctly load the CSV file
    df = pd.read_csv(p)

    # Extract month and year from week_number and create a new column 'month_year'
    def extract_month_year(week_str):
        try:
            week, year = map(int, week_str.split('_'))
            # This is a simplified approach. For precise ISO week date to month,
            # a more robust library or calculation might be needed.
            # Assuming week 38, 39 -> Sep, week 42, 44 -> Oct for 2025 as per user request.
            if year == 2025:
                if week in [38, 39]:
                    return '09_2025' # September 2025
                elif week in [42, 44]:
                    return '10_2025' # October 2025
                else:
                    return '11_2025'
            # Default to a generic format if not in specified range
            return f'{week:02d}_{year}'
        except (ValueError, IndexError):
            return None # Handle unexpected formats

    df['month_year'] = df['week_number'].apply(extract_month_year)
    df['month_year_formatted'] = df['month_year'].apply(format_month_year)

    # Replace None or NaN, 'Other', 'Others' in 'Sector' with empty string for display
    df['Sector'] = df['Sector'].replace([None, 'None', np.nan, 'Other', 'Others'], '')

    # Drop duplicates based on the 'Summary' column
    df.drop_duplicates(subset=['Summary'], inplace=True)

    return df


summaries = load_summaries(PROCESSED_SUMMARIES_PATH)

# Calculate average sentiment scores
@st.cache_data
def calculate_average_sentiment(df):
    # Average sentiment score by country and asset class, grouped by month_year
    avg_country_asset = df.groupby(['month_year', 'Countries_Region', 'Asset_Class'])['Score'].mean().reset_index()
    avg_country_asset.rename(columns={'Score': 'Avg_Score'}, inplace=True)

    # Average sentiment score by country, asset class, and sector, grouped by month_year
    avg_country_asset_sector = df.groupby(['month_year', 'Countries_Region', 'Asset_Class', 'Sector'])['Score'].mean().reset_index()
    avg_country_asset_sector.rename(columns={'Score': 'Avg_Score'}, inplace=True)

    return avg_country_asset, avg_country_asset_sector


average_sentiment_country_asset, average_sentiment_country_asset_sector = calculate_average_sentiment(summaries)


# ════════════════════ Navigation ═══════════════════
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Go to", ["Documents Processed", "Sentiment Analysis", "Summary Example", "UOB vs Others"], label_visibility="collapsed")


# ════════════════════ Documents Processed Page ═══════════════════
if page == "Documents Processed":
    st.markdown("# Documents Processed")

    # Calculate total DISTINCT analyst reports captured per month
    total_reports_per_month = summaries.groupby('month_year_formatted')['Filename'].nunique().reset_index()
    total_reports_per_month.rename(columns={'Filename': 'Total_Analyst_Reports'}, inplace=True)

    # Calculate number of distinct financial institutes per month
    distinct_institutes_per_month = summaries.groupby('month_year_formatted')['Financial_Institute'].nunique().reset_index()
    distinct_institutes_per_month.rename(columns={'Financial_Institute': 'Distinct_Financial_Institutes'}, inplace=True)

    # Merge the counts
    monthly_summary = total_reports_per_month.merge(distinct_institutes_per_month, on='month_year_formatted')

    st.markdown("### Monthly Summary")
    # Display the monthly summary
    st.dataframe(monthly_summary[['month_year_formatted', 'Total_Analyst_Reports', 'Distinct_Financial_Institutes']], hide_index=True)

    # Group by month_year and financial institute and count distinct filenames
    file_counts = summaries.groupby(['month_year_formatted', 'Financial_Institute'])['Filename'].nunique().reset_index()
    file_counts.rename(columns={'Filename': 'Distinct_Filename_Count'}, inplace=True)

    st.markdown("### By Financial Institute")
    # Display the file counts
    st.dataframe(file_counts[['month_year_formatted','Financial_Institute', 'Distinct_Filename_Count']], hide_index=True)


# ════════════════════ Summary Example Page ═══════════════════
elif page == "Summary Example":
    st.markdown("# Asset Class Sentiment Analysis")

    # Dropdown filters in the sidebar
    with st.sidebar:
        st.markdown("### Filters")
        all_institutes = summaries['Financial_Institute'].unique().tolist()
        # Changed to single select
        selected_institute = st.selectbox("Select Financial Institute", all_institutes, key='institute_filter')

        # Filter countries/regions based on selected institute
        filtered_countries_regions = summaries[
            (summaries['Financial_Institute'] == selected_institute)
        ]['Countries_Region'].unique().tolist()
        selected_country_region = st.selectbox("Select Country/Region", filtered_countries_regions, key='country_region_filter')

        # Filter asset classes based on selected institute and country/region
        filtered_asset_classes = summaries[
            (summaries['Financial_Institute'] == selected_institute) &
            (summaries['Countries_Region'] == selected_country_region)
        ]['Asset_Class'].unique().tolist()
        selected_class = st.selectbox("Select Asset Class", filtered_asset_classes, key='asset_class_filter')

        # Add month_year filter using formatted month years
        all_month_years_formatted = summaries['month_year_formatted'].unique().tolist()
        selected_month_year_formatted = st.selectbox("Select Month and Year", all_month_years_formatted, key='month_year_filter')

        # Map selected formatted month year back to original month_year for filtering
        selected_month_year = summaries[summaries['month_year_formatted'] == selected_month_year_formatted]['month_year'].iloc[0]


    # Filtered DataFrame
    filtered_df = summaries[
        (summaries['Financial_Institute'] == selected_institute) & # Updated filtering
        (summaries['Asset_Class'] == selected_class) &
        (summaries['Countries_Region'] == selected_country_region) &
        (summaries['month_year'] == selected_month_year) # Apply month_year filter
    ].copy() # Create a copy to avoid SettingWithCopyWarning

    # Replace None or NaN in 'Sector' with empty string
    filtered_df['Sector'] = filtered_df['Sector'].replace([None, 'None', np.nan, 'Other', 'Others'], '')


    # Calculate and display average score
    if not filtered_df.empty:
        # Calculate the average of the average score per financial institution
        avg_score_per_institute = filtered_df.groupby('Financial_Institute')['Score'].mean()
        average_score = avg_score_per_institute.mean()
        st.info(f"**Average Sentiment Score:** {average_score:.2f}")

        # Display results in a table
        st.markdown(f"### Filtered Summaries for {selected_country_region} - {selected_class} ({selected_month_year_formatted})")

        # Define the desired column order
        display_columns = ['Financial_Institute', 'Countries_Region', 'Asset_Class', 'Sector', 'Summary', 'Sentiment']

        # Using st.write with to_html to better control display for non-wrapping
        st.write(format_dataframe_for_display(filtered_df, display_columns).to_html(index=False), unsafe_allow_html=True)


    else:
        st.info("No summaries found for the selected filters.")

# ════════════════════ Sentiment Analysis Results Page ═══════════════════
elif page == "Sentiment Analysis":
    st.markdown("# Sentiment Analysis Summary")

    # Add month_year filter to the sidebar
    with st.sidebar:
        st.markdown("### Sentiment Analysis Filters")
        all_month_years_formatted_results = summaries['month_year_formatted'].unique().tolist()
        selected_month_year_formatted_results = st.selectbox("Select Month and Year", all_month_years_formatted_results, key='month_year_results_filter_sidebar')

        # Map selected formatted month year back to original month_year for filtering
        selected_month_year_results = summaries[summaries['month_year_formatted'] == selected_month_year_formatted_results]['month_year'].iloc[0]

    # Filter summaries based on the selected month_year
    filtered_summaries_results = summaries[
        (summaries['month_year'] == selected_month_year_results) &
        (summaries['Score'].notna())
    ].copy()

    # Calculate average sentiment scores and count unique financial institutes
    # New two-step aggregation for sentiment_counts
    avg_score_per_institute_country_asset = filtered_summaries_results.groupby(['Countries_Region', 'Asset_Class', 'Financial_Institute'])['Score'].mean().reset_index()
    sentiment_counts = avg_score_per_institute_country_asset.groupby(['Countries_Region', 'Asset_Class'])['Score'].agg(['mean', 'count']).reset_index()
    sentiment_counts.rename(columns={'mean': 'Avg_Score', 'count': 'Record_Count'}, inplace=True)

    # Calculate unique financial institute count for each group
    unique_institutes_count = filtered_summaries_results.groupby(['Countries_Region', 'Asset_Class'])['Financial_Institute'].nunique().reset_index()
    unique_institutes_count.rename(columns={'Financial_Institute': 'Unique_Institute_Count'}, inplace=True)

    # Merge the average sentiment and unique institute counts
    sentiment_results = pd.merge(sentiment_counts, unique_institutes_count, on=['Countries_Region', 'Asset_Class'])

    # Filter for entries with at least two unique financial institutes
    filtered_sentiment_results = sentiment_results[sentiment_results['Unique_Institute_Count'] >= 2].copy()

    # Calculate average sentiment scores for the filtered data by sector for the detail tab
    # New two-step aggregation for average_sentiment_country_asset_sector_results
    avg_score_per_institute_country_asset_sector = filtered_summaries_results.groupby(['Countries_Region', 'Asset_Class', 'Sector', 'Financial_Institute'])['Score'].mean().reset_index()
    average_sentiment_country_asset_sector_results = avg_score_per_institute_country_asset_sector.groupby(['Countries_Region', 'Asset_Class', 'Sector'])['Score'].mean().reset_index()
    average_sentiment_country_asset_sector_results.rename(columns={'Score': 'Avg_Score'}, inplace=True)


    # Filter data into three categories
    positive_sentiment = filtered_sentiment_results[filtered_sentiment_results['Avg_Score'] > 0.3].sort_values('Avg_Score', ascending=False)
    neutral_sentiment = filtered_sentiment_results[(filtered_sentiment_results['Avg_Score'] <= 0.3) & (filtered_sentiment_results['Avg_Score'] >= -0.3)].sort_values('Avg_Score', ascending=False)
    negative_sentiment = filtered_sentiment_results[filtered_sentiment_results['Avg_Score'] < -0.3].sort_values('Avg_Score', ascending=False)

    # Display in three columns
    col1, col2, col3 = st.columns(3)

    selected_row = None

    with col1:
        st.markdown("### Positive (Score > 0.3)")
        if not positive_sentiment.empty:
            # Display full rows and prevent sliding
            selected_row_data = st.dataframe(positive_sentiment[['Countries_Region', 'Asset_Class', 'Avg_Score']].style.format({'Avg_Score': '{:.2f}'}), hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row")
            if selected_row_data and selected_row_data["selection"]["rows"]:
                selected_row_index = selected_row_data["selection"]["rows"][0]
                selected_row = positive_sentiment.iloc[selected_row_index]

        else:
            st.info("No data for positive sentiment from at least two institutions.")

    with col2:
        st.markdown("### Neutral (-0.3 <= Score <= 0.3)")
        if not neutral_sentiment.empty:
             # Display full rows and prevent sliding
            selected_row_data = st.dataframe(neutral_sentiment[['Countries_Region', 'Asset_Class', 'Avg_Score']].style.format({'Avg_Score': '{:.2f}'}), hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row")
            if selected_row_data and selected_row_data["selection"]["rows"]:
                selected_row_index = selected_row_data["selection"]["rows"][0]
                selected_row = neutral_sentiment.iloc[selected_row_index]
        else:
            st.info("No data for neutral sentiment from at least two institutions.")

    with col3:
        st.markdown("### Negative (Score < -0.3)")
        if not negative_sentiment.empty:
             # Display full rows and prevent sliding
            selected_row_data = st.dataframe(negative_sentiment[['Countries_Region', 'Asset_Class', 'Avg_Score']].style.format({'Avg_Score': '{:.2f}'}), hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row")
            if selected_row_data and selected_row_data["selection"]["rows"]:
                selected_row_index = selected_row_data["selection"]["rows"][0]
                selected_row = negative_sentiment.iloc[selected_row_index]
        else:
            st.info("No data for negative sentiment from at least two institutions.")

    # Display detailed sentiment by sector and summaries for the selected row in tabs
    if selected_row is not None:
        st.markdown(f"### Details for {selected_row['Countries_Region']} - {selected_row['Asset_Class']} ({selected_month_year_formatted_results})") # Add month_year_formatted
        tab1, tab2 = st.tabs(["Sector Sentiment", "Summaries"])

        with tab1:
            st.markdown("### Sector Sentiment (High to Low)")
            sector_sentiment_filtered = average_sentiment_country_asset_sector_results[
                (average_sentiment_country_asset_sector_results['Countries_Region'] == selected_row['Countries_Region']) &
                (average_sentiment_country_asset_sector_results['Asset_Class'] == selected_row['Asset_Class'])
            ].sort_values('Avg_Score', ascending=False) # Sort by sentiment score

            # Filter for entries with at least two unique financial institutes at the sector level
            sector_unique_institutes = filtered_summaries_results.groupby(['Countries_Region', 'Asset_Class', 'Sector'])['Financial_Institute'].nunique().reset_index()
            sector_unique_institutes.rename(columns={'Financial_Institute': 'Sector_Unique_Institute_Count'}, inplace=True)

            # Removed the filtering based on unique institute count for sector sentiment
            # sector_sentiment_filtered = pd.merge(sector_sentiment_filtered, sector_unique_institutes, on=['Countries_Region', 'Asset_Class', 'Sector'])
            # sector_sentiment_filtered = sector_sentiment_filtered[sector_sentiment_filtered['Sector_Unique_Institute_Count'] >= 2].copy()


            if not sector_sentiment_filtered.empty:
                st.dataframe(sector_sentiment_filtered[['Countries_Region', 'Asset_Class', 'Sector', 'Avg_Score']].style.format({'Avg_Score': '{:.2f}'}), hide_index=True, use_container_width=True)
            else:
                st.info("No sector-specific sentiment data available for this selection.")

        with tab2:
            st.markdown("### Relevant Summaries")
            related_summaries = filtered_summaries_results[
                (filtered_summaries_results['Countries_Region'] == selected_row['Countries_Region']) &
                (filtered_summaries_results['Asset_Class'] == selected_row['Asset_Class'])
            ]
            if not related_summaries.empty:
                st.write(format_dataframe_for_display(related_summaries, ['Financial_Institute', 'Countries_Region', 'Asset_Class', 'Sector', 'Summary', 'Sentiment']).to_html(index=False), unsafe_allow_html=True)
            else:
                st.info("No summaries found for this selection.")


# ════════════════════ UOB vs Others Sentiment Page ═══════════════════
elif page == "UOB vs Others":
    st.markdown("# UOB vs Other Financial Institutions")

    with st.sidebar:
        st.markdown("### Comparison Filters")
        all_month_years_formatted_comparison = summaries['month_year_formatted'].unique().tolist()
        selected_month_year_formatted_comparison = st.selectbox("Select Month and Year", all_month_years_formatted_comparison, key='month_year_comparison_filter')

        selected_month_year_comparison = summaries[summaries['month_year_formatted'] == selected_month_year_formatted_comparison]['month_year'].iloc[0]

    filtered_summaries_comparison = summaries[
        (summaries['month_year'] == selected_month_year_comparison)
    ].copy()

    # Filter out rows where Countries_Region is 'Global' and Sector is empty
    filtered_summaries_comparison = filtered_summaries_comparison[
        ~((filtered_summaries_comparison['Countries_Region'] == 'Global') & (filtered_summaries_comparison['Sector'] == ''))
    ].copy()


    if not filtered_summaries_comparison.empty:
        # Separate UOB data and Others data
        uob_df = filtered_summaries_comparison[filtered_summaries_comparison['Financial_Institute'] == 'UOB']
        others_df = filtered_summaries_comparison[filtered_summaries_comparison['Financial_Institute'] != 'UOB']

        # Group by country, asset class, AND sector and calculate average sentiment
        uob_sentiment = uob_df.groupby(['Countries_Region', 'Asset_Class', 'Sector'])['Score'].mean().reset_index()
        uob_sentiment.rename(columns={'Score': 'UOB'}, inplace=True)

        others_sentiment = others_df.groupby(['Countries_Region', 'Asset_Class', 'Sector'])['Score'].mean().reset_index()
        others_sentiment.rename(columns={'Score': 'Others'}, inplace=True)

        # Merge the sentiment dataframes on all three columns
        comparison_df = pd.merge(uob_sentiment, others_sentiment, on=['Countries_Region', 'Asset_Class', 'Sector'], how='outer')

        # Filter for asset classes where either UOB or Others have expressed sentiment (Score is not NaN)
        comparison_df_filtered = comparison_df.dropna(subset=['UOB', 'Others'], how='any').copy()

        if not comparison_df_filtered.empty:
            st.markdown(f"### {selected_month_year_formatted_comparison}")

            # Function to apply highlighting
            def highlight_sentiment_difference(row):
                color = ''
                uob_score = row['UOB']
                others_score = row['Others']

                if pd.notna(uob_score) and pd.notna(others_score):
                    sentiment_diff = abs(uob_score - others_score)

                    # Highlight if the absolute difference is greater than 0.5 (prioritize red)
                    if sentiment_diff > 0.5:
                         color = 'background-color: red'
                    # Highlight if one is positive and the other is negative (yellow)
                    elif (uob_score > 0 and others_score < 0) or (uob_score < 0 and others_score > 0):
                         color = 'background-color: yellow'

                # Highlight if only one has sentiment (yellow)
                elif pd.notna(uob_score) or pd.notna(others_score):
                    color = 'background-color: yellow'

                return [color] * len(row)

            # Separate data into 'Similar Sentiment' and 'Others' based on criteria
            similar_sentiment_df = comparison_df_filtered[
                ((np.sign(comparison_df_filtered['UOB']) == np.sign(comparison_df_filtered['Others'])) &
                 (abs(comparison_df_filtered['UOB'] - comparison_df_filtered['Others']) < 0.8)) |
                 ((comparison_df_filtered['UOB'] * comparison_df_filtered['Others'] == 0) & (abs(comparison_df_filtered['UOB'] - comparison_df_filtered['Others']) < 0.3))
            ].copy()

            others_sentiment_df = comparison_df_filtered[
                ~(((np.sign(comparison_df_filtered['UOB']) == np.sign(comparison_df_filtered['Others'])) &
                 (abs(comparison_df_filtered['UOB'] - comparison_df_filtered['Others']) < 0.8)) |
                 ((comparison_df_filtered['UOB'] * comparison_df_filtered['Others'] == 0) & (abs(comparison_df_filtered['UOB'] - comparison_df_filtered['Others']) < 0.3)))
            ].copy()


            st.markdown("### Similar Sentiment (Same Sign or Difference < 0.3)")
            if not similar_sentiment_df.empty:
                 # Display full rows and prevent sliding
                 st.dataframe(similar_sentiment_df.style.apply(highlight_sentiment_difference, axis=1).format({
                    'UOB': '{:.2f}',
                    'Others': '{:.2f}'
                }).data, hide_index=True, use_container_width=True)
            else:
                st.info("No entries with similar sentiment for the selected filters.")

            st.markdown("### Different Sentiments")
            if not others_sentiment_df.empty:
                 # Display full rows and prevent sliding
                 st.dataframe(others_sentiment_df.style.apply(highlight_sentiment_difference, axis=1).format({
                    'UOB': '{:.2f}',
                    'Others': '{:.2f}'
                }).data, hide_index=True, use_container_width=True)
            else:
                st.info("No other entries for the selected filters.")

        else:
            st.info("No asset classes with sentiment expressed by UOB or other institutions for the selected filters.")

    else:
        st.info("No data available for the selected filters.")
