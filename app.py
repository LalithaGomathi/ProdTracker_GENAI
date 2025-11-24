import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
from datetime import datetime
import altair as alt

st.set_page_config(page_title="Productivity Tracker", layout="wide")
st.title("Productivity Tracker — Demo (Streamlit-only)")

with st.expander("About this app"):
    st.write("Upload ticket and call CSVs. Computes productive time, utilization, AHT, idle time, and shows a day/hour heatmap (Option A).")

col1, col2 = st.columns([1,2])
with col1:
    tickets_file = st.file_uploader("Tickets CSV", type="csv")
    calls_file = st.file_uploader("Calls CSV", type="csv")
    if st.button("Use sample data"):
        tickets_file = open("sample_data/tickets.csv","rb")
        calls_file = open("sample_data/calls.csv","rb")
    start_date = st.date_input("Start date", value=(datetime.utcnow() - pd.Timedelta(days=7)).date())
    end_date = st.date_input("End date", value=datetime.utcnow().date())
    default_shift_hours = st.number_input("Default shift hours", min_value=1, max_value=24, value=8)
    overlap_mode = st.selectbox("Overlap handling", options=["split","full"])
with col2:
    st.header("Quick info")
    st.write("Heatmap: day rows, hour columns.")

def parse_csv_like(f):
    if f is None:
        return None
    if hasattr(f, "read"):
        raw = f.read()
        if isinstance(raw, bytes):
            raw = raw.decode()
        return pd.read_csv(StringIO(raw), parse_dates=['start_time','end_time'])
    return pd.read_csv(f, parse_dates=['start_time','end_time'])

def clip_to_period(df, start_p, end_p):
    df = df.copy()
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    df['start_time_clipped'] = df['start_time'].clip(lower=start_p, upper=end_p)
    df['end_time_clipped'] = df['end_time'].clip(lower=start_p, upper=end_p)
    df = df[df['end_time_clipped'] > df['start_time_clipped']]
    return df

def merge_intervals(intervals):
    if not intervals:
        return []
    ints = sorted(intervals, key=lambda x: x[0])
    merged = [list(ints[0])]
    for s,e in ints[1:]:
        if s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s,e])
    return merged

def compute_metrics(events, start_p, end_p, overlap_mode, default_shift_hours):
    events = events.copy()
    events['duration_s'] = (events['end_time_clipped'] - events['start_time_clipped']).dt.total_seconds()
    agents = events['agent'].unique().tolist()
    business_days = pd.bdate_range(start=start_p.date(), end=end_p.date()).size
    scheduled_seconds = business_days * default_shift_hours * 3600
    results = {}
    for a in agents:
        adf = events[events['agent']==a]
        if overlap_mode == 'full':
            productive = adf['duration_s'].sum()
        else:
            intervals = list(zip(adf['start_time_clipped'], adf['end_time_clipped']))
            merged = merge_intervals(intervals)
            productive = sum((e - s).total_seconds() for s,e in merged)
        utilization = productive / scheduled_seconds if scheduled_seconds>0 else 0
        cat_avg = adf.groupby('category')['duration_s'].mean().to_dict()
        results[a] = {
            'productive_seconds': int(productive),
            'scheduled_seconds': int(scheduled_seconds),
            'utilization': round(utilization,4),
            'avg_handle_time_by_category_seconds': {k:int(v) for k,v in cat_avg.items()}
        }
    return results

def build_heatmap(events, start_p, end_p):
    records = []
    for _, row in events.iterrows():
        s = row['start_time_clipped']
        e = row['end_time_clipped']
        while s < e:
            bucket_end = (s + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            if bucket_end <= s:
                bucket_end = s + pd.Timedelta(hours=1)
            part_end = min(e, bucket_end)
            records.append({'date': s.date(), 'hour': s.hour, 'seconds': (part_end - s).total_seconds()})
            s = part_end
    if not records:
        return None
    rdf = pd.DataFrame(records)
    heat = rdf.groupby(['date','hour'])['seconds'].sum().reset_index()
    heat['hours'] = heat['seconds']/3600.0
    return heat

if st.button("Process"):
    start_p = pd.to_datetime(str(start_date) + "T00:00:00")
    end_p = pd.to_datetime(str(end_date) + "T23:59:59")
    dfs = []
    if tickets_file is not None:
        try:
            tdf = parse_csv_like(tickets_file)
            tdf = clip_to_period(tdf, start_p, end_p)
            tdf['category'] = tdf.get('category','ticket').fillna('ticket')
            dfs.append(tdf)
        except Exception as e:
            st.error("Failed to parse tickets: " + str(e))
            st.stop()
    if calls_file is not None:
        try:
            cdf = parse_csv_like(calls_file)
            cdf = clip_to_period(cdf, start_p, end_p)
            cdf['category'] = cdf.get('category','call').fillna('call')
            dfs.append(cdf)
        except Exception as e:
            st.error("Failed to parse calls: " + str(e))
            st.stop()
    if not dfs:
        st.error("Upload at least one file or use sample data.")
        st.stop()
    events = pd.concat(dfs, ignore_index=True, sort=False)
    results = compute_metrics(events, start_p, end_p, overlap_mode, default_shift_hours)
    total_prod = sum(r['productive_seconds'] for r in results.values())
    total_sched = sum(r['scheduled_seconds'] for r in results.values())
    avg_util = np.mean([r['utilization'] for r in results.values()]) if results else 0
    st.metric("Total productive (h)", f"{total_prod/3600:.2f}")
    st.metric("Total scheduled (h)", f"{total_sched/3600:.2f}")
    st.metric("Avg utilization (%)", f"{avg_util*100:.1f}")
    st.header("Per-agent summary")
    rows = []
    for a,r in results.items():
        rows.append({'agent': a,
                      'productive_h': r['productive_seconds']/3600,
                      'scheduled_h': r['scheduled_seconds']/3600,
                      'util_percent': r['utilization']*100,
                      'idle_h': max(0,(r['scheduled_seconds']-r['productive_seconds'])/3600)})
    df_report = pd.DataFrame(rows)
    st.dataframe(df_report)
    st.header("Avg handle time by category (sec)")
    cat_rows = []
    for a,r in results.items():
        for cat,sec in r['avg_handle_time_by_category_seconds'].items():
            cat_rows.append({'agent': a, 'category': cat, 'avg_seconds': sec})
    if cat_rows:
        st.table(pd.DataFrame(cat_rows))
    st.header("Day/Hour heatmap (hours)")
    heat = build_heatmap(events, start_p, end_p)
    if heat is None or heat.empty:
        st.info("No events to show heatmap.")
    else:
        chart = alt.Chart(heat).mark_rect().encode(
            x=alt.X('hour:O', title='Hour'),
            y=alt.Y('date:O', title='Date'),
            color=alt.Color('hours:Q', title='Hours', scale=alt.Scale(scheme='greens'))
        ).properties(width=900, height=300)
        st.altair_chart(chart, use_container_width=True)
    csv_bytes = df_report.to_csv(index=False).encode('utf-8')
    st.download_button("Export per-agent CSV", csv_bytes, file_name="per_agent_report.csv", mime='text/csv')
