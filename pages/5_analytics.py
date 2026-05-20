"""
pages/5_analytics.py — EDA and analytics charts from real training data.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline import load_data
import config

st.title("📈 Analytics")
st.caption("Exploratory data analysis from the training dataset.")
st.divider()

@st.cache_data(show_spinner="Loading data…")
def get_raw():
    train_raw, test_raw, zone_summary, _ = load_data()
    # Combine for analytics
    full = pd.concat([train_raw, test_raw], ignore_index=True)
    for col in ['Month_Number','Year','Actual_Demand (cylinders)',
                'Stockout_Occurred','Zone_Opening_Stock',
                'Zone_Closing_Stock','Monthly_Income (₹)',
                'LPG_Price_per_Cylinder (₹)']:
        if col in full.columns:
            full[col] = pd.to_numeric(full[col], errors='coerce')
    return full, zone_summary

try:
    df, zone_summary = get_raw()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
SEASON_COLORS = {
    1:'#5DADE2',2:'#5DADE2',3:'#F39C12',4:'#F39C12',5:'#F39C12',
    6:'#27AE60',7:'#27AE60',8:'#27AE60',9:'#27AE60',
    10:'#E67E22',11:'#E67E22',12:'#5DADE2',
}
COLORS = ['#1A5276','#1E8449','#B7950B','#C0392B',
          '#7D3C98','#117A65','#B03A2E','#1F618D','#196F3D','#784212']

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Monthly Trends", "🏭 Zone Analysis",
    "📊 Demand Distribution", "📈 Year-on-Year", "🌡️ Heatmap"
])

# ─── TAB 1: Monthly Trends ────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Average demand by month**")
        monthly = (df.groupby('Month_Number')['Actual_Demand (cylinders)']
                   .mean().reset_index())
        monthly['Month_Name'] = monthly['Month_Number'].apply(lambda x: MONTH_NAMES[x-1])
        bar_colors = [SEASON_COLORS[m] for m in monthly['Month_Number']]

        fig, ax = plt.subplots(figsize=(7,4))
        bars = ax.bar(monthly['Month_Name'], monthly['Actual_Demand (cylinders)'],
                      color=bar_colors, edgecolor='white', width=0.6)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)
        legend_els = [
            mpatches.Patch(color='#5DADE2', label='Winter (Dec–Feb)'),
            mpatches.Patch(color='#F39C12', label='Summer (Mar–May)'),
            mpatches.Patch(color='#27AE60', label='Monsoon (Jun–Sep)'),
            mpatches.Patch(color='#E67E22', label='Autumn (Oct–Nov)'),
        ]
        ax.legend(handles=legend_els, fontsize=8)
        ax.set_ylabel('Avg demand (cylinders)')
        ax.set_title('Monthly avg LPG demand')
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("**Stockout rate by month**")
        so_monthly = (df.groupby('Month_Number')['Stockout_Occurred']
                      .mean().reset_index())
        so_monthly['Month_Name'] = so_monthly['Month_Number'].apply(lambda x: MONTH_NAMES[x-1])
        so_colors = ['#C0392B' if v > so_monthly['Stockout_Occurred'].mean()
                     else '#1E8449' for v in so_monthly['Stockout_Occurred']]

        fig, ax = plt.subplots(figsize=(7,4))
        bars = ax.bar(so_monthly['Month_Name'], so_monthly['Stockout_Occurred']*100,
                      color=so_colors, edgecolor='white', width=0.6)
        ax.axhline(so_monthly['Stockout_Occurred'].mean()*100,
                   color='orange', linestyle='--', linewidth=1.5, label='Avg')
        ax.set_ylabel('Stockout rate (%)')
        ax.set_title('Monthly stockout rate')
        ax.legend(fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

# ─── TAB 2: Zone Analysis ─────────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Average demand by zone**")
        zone_demand = (df.groupby('Warehouse_Zone')['Actual_Demand (cylinders)']
                       .mean().sort_values(ascending=False))
        fig, ax = plt.subplots(figsize=(6,5))
        bars = ax.barh(zone_demand.index, zone_demand.values,
                       color=COLORS[:len(zone_demand)], height=0.6)
        for bar, val in zip(bars, zone_demand.values):
            ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=9)
        ax.set_xlabel('Avg demand (cylinders)')
        ax.set_title('Zone-wise avg demand')
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("**Stockout rate by zone**")
        zone_so = (df.groupby('Warehouse_Zone')['Stockout_Occurred']
                   .mean().sort_values(ascending=False) * 100)
        so_colors_zone = ['#C0392B' if v > zone_so.mean() else '#1E8449'
                          for v in zone_so.values]
        fig, ax = plt.subplots(figsize=(6,5))
        bars = ax.bar(zone_so.index, zone_so.values,
                      color=so_colors_zone, edgecolor='white', width=0.6)
        ax.axhline(zone_so.mean(), color='orange', linestyle='--',
                   linewidth=1.5, label=f'Avg: {zone_so.mean():.1f}%')
        ax.set_ylabel('Stockout rate (%)')
        ax.set_title('Stockout rate by zone (red = above avg)')
        ax.set_xticklabels(zone_so.index, rotation=30, ha='right')
        ax.legend(fontsize=9)
        for bar, val in zip(bars, zone_so.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("**Zone Summary Table**")
    if 'Warehouse_Zone' in zone_summary.columns:
        st.dataframe(zone_summary.set_index('Warehouse_Zone'), use_container_width=True)

# ─── TAB 3: Demand Distribution ───────────────────────────────────────────────
with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Overall demand distribution**")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(df['Actual_Demand (cylinders)'].dropna(), bins=40,
                color='#1A5276', alpha=0.75, edgecolor='white')
        ax.set_xlabel('Demand (cylinders)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of actual demand')
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("**Demand by subsidy type**")
        sub_demand = df.groupby('Subsidy_Type')['Actual_Demand (cylinders)'].mean()
        fig, ax = plt.subplots(figsize=(5,4))
        ax.bar(sub_demand.index, sub_demand.values,
               color=['#1A5276','#1E8449'], edgecolor='white', width=0.4)
        ax.set_ylabel('Avg demand (cylinders)')
        ax.set_title('Avg demand by subsidy type')
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("**Demand vs Monthly Income**")
    sample = df[['Monthly_Income (₹)','Actual_Demand (cylinders)']].dropna().sample(
        min(500, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.scatter(sample['Monthly_Income (₹)'], sample['Actual_Demand (cylinders)'],
               alpha=0.3, s=12, color='#1A5276')
    ax.set_xlabel('Monthly income (₹)')
    ax.set_ylabel('Actual demand (cylinders)')
    ax.set_title('Demand vs income (sample of 500 records)')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

# ─── TAB 4: Year-on-Year ──────────────────────────────────────────────────────
with tab4:
    st.markdown("**Year-on-Year demand trend**")
    yoy = df.groupby(['Year','Month_Number'])['Actual_Demand (cylinders)'].mean().reset_index()
    years = sorted(yoy['Year'].dropna().unique())

    fig, ax = plt.subplots(figsize=(11,5))
    for i, yr in enumerate(years):
        subset = yoy[yoy['Year']==yr].sort_values('Month_Number')
        ax.plot(subset['Month_Number'], subset['Actual_Demand (cylinders)'],
                marker='o', markersize=4, linewidth=1.8,
                color=COLORS[i % len(COLORS)], label=str(int(yr)))
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_ylabel('Avg demand (cylinders)')
    ax.set_title('Year-on-Year LPG demand trend by month')
    ax.legend(title='Year', fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("**Annual totals**")
    annual = df.groupby('Year')['Actual_Demand (cylinders)'].agg(['sum','mean','count'])
    annual.columns = ['Total Demand','Avg Demand','Records']
    annual.index = annual.index.astype(int)
    st.dataframe(annual.style.format({'Total Demand':'{:.0f}','Avg Demand':'{:.3f}'}),
                 use_container_width=True)

# ─── TAB 5: Heatmap ──────────────────────────────────────────────────────────
with tab5:
    st.markdown("**Demand heatmap — Zone × Month**")
    pivot = df.pivot_table(
        values='Actual_Demand (cylinders)',
        index='Warehouse_Zone',
        columns='Month_Number',
        aggfunc='mean',
    )
    pivot.columns = [MONTH_NAMES[m-1] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(12,5))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                linewidths=0.3, cbar_kws={'label':'Avg demand (cylinders)'})
    ax.set_title('Avg LPG demand — Zone × Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Warehouse Zone')
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("**Stockout heatmap — Zone × Month**")
    pivot_so = df.pivot_table(
        values='Stockout_Occurred',
        index='Warehouse_Zone',
        columns='Month_Number',
        aggfunc='mean',
    ) * 100
    pivot_so.columns = [MONTH_NAMES[m-1] for m in pivot_so.columns]

    fig, ax = plt.subplots(figsize=(12,5))
    sns.heatmap(pivot_so, annot=True, fmt='.1f', cmap='Reds', ax=ax,
                linewidths=0.3, cbar_kws={'label':'Stockout rate (%)'})
    ax.set_title('Stockout rate (%) — Zone × Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Warehouse Zone')
    fig.tight_layout()
    st.pyplot(fig); plt.close()
