import streamlit as st
import requests
import pandas as pd
import base64
from datetime import date, timedelta, datetime
from io import BytesIO
from fpdf import FPDF
import plotly.express as px
from pathlib import Path
from dotenv import load_dotenv
import os

# 🔐 Manually specify the path to your .env file
env_path = Path("C:/Users/PC/Documents/UEA/.env")
load_dotenv(dotenv_path=env_path)

# ✅ Now read the values
API_KEY = os.getenv("API_KEY")
API_TOKEN = os.getenv("API_TOKEN")

# 🔍 Optional: Debug print to verify values are loaded (remove after testing)
print("🔍 API_KEY loaded:", repr(API_KEY))
print("🔍 API_TOKEN loaded:", repr(API_TOKEN))




def get_auth_header():
    token = f"{API_KEY}:{API_TOKEN}"
    b64 = base64.b64encode(token.encode()).decode()
    return {"Authorization": f"Basic {b64}", "Accept": "application/json"}

def fetch_bookings(start_date, end_date):
    url = "https://theshoebox-dev.checkfront.co.uk/api/3.0/booking"
    headers = get_auth_header()
    params = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "limit": 500
    }
    res = requests.get(url, headers=headers, params=params)
    res.raise_for_status()
    return res.json()

def fetch_booking_details(booking_id):
 url = f"https://theshoebox-dev.checkfront.co.uk/api/3.0/booking/{booking_id}"
 headers = get_auth_header()
 res = requests.get(url, headers=headers)
 res.raise_for_status()
 return res.json()

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Bookings')
    return output.getvalue()

def create_detailed_pdf_summary(kpi_data, date_range, top_tour, top_day, recent_rows, logo_path=None):
    pdf = FPDF()
    pdf.add_page()

    if logo_path and Path(logo_path).exists():
        pdf.image(str(logo_path), x=10, y=8, w=33)
        pdf.set_xy(50, 10)
    else:
        pdf.set_y(15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Shoebox Sales Summary Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Date Range: {date_range}", ln=True)
    pdf.ln(2)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Key Metrics", ln=True)
    pdf.set_font("Arial", "", 12)
    for label, value in kpi_data.items():
        pdf.cell(0, 8, f"{label}: {value}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Top Performer", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Best-Selling Tour: {top_tour}", ln=True)
    pdf.cell(0, 8, f"Most Popular Day: {top_day}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Recent Bookings", ln=True)
    pdf.set_font("Arial", "", 11)

    col_widths = [15, 50, 25, 20, 30]
    headers = ["#", "Customer", "Amount", "Status", "Date"]
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, border=1)
    pdf.ln()

    for row in recent_rows:
        pdf.cell(col_widths[0], 8, str(row["booking_id"]), border=1)
        pdf.cell(col_widths[1], 8, str(row["customer_name"])[:24], border=1)
        pdf.cell(col_widths[2], 8, f"£{row['total']:.2f}", border=1)
        pdf.cell(col_widths[3], 8, row["status_name"], border=1)
        date_str = datetime.strftime(row["created_date"], "%Y-%m-%d")
        pdf.cell(col_widths[4], 8, date_str, border=1)
        pdf.ln()

    return pdf.output(dest="S").encode("latin-1")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Shoebox Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'> Shoebox Internal Operations Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    logo = Path(__file__).parent / "shoebox.png"
    if logo.exists():
        st.image(str(logo), width=180)
    else:
        st.warning("Logo not found")
    
    today = date.today()
    start = st.date_input(" Start Date", today - timedelta(days=30))
    end = st.date_input(" End Date", today + timedelta(days=60))
    search = st.text_input("🔍 Search name, email or booking code").lower()

    try:
        temp_raw = fetch_bookings(start, end)
        temp_df = pd.DataFrame(temp_raw.get("booking/index", {}).values())
        status_options = ["All"] + sorted(temp_df["status_name"].dropna().unique()) if not temp_df.empty else ["All"]
    except:
        status_options = ["All"]

    status_filter = st.selectbox(" Filter by Booking Status", status_options)
    
    if st.button("🔄 Refresh"):
        st.rerun()

# --- MAIN DASHBOARD ---
try:
    raw = fetch_bookings(start, end)
    bookings = list(raw.get("booking/index", {}).values())
    df = pd.DataFrame(bookings)
    
    # Add ticket quantity column
    df["ticket_qty"] = 0

# Safely fetch ticket quantities for each booking
    for i, row in df.iterrows():
     try:
        details = fetch_booking_details(row["booking_id"])
        booking_info = details.get("booking", {})
        items = booking_info.get("items", [])

        if isinstance(items, dict):
            items = list(items.values())
        elif not isinstance(items, list):
            items = []

        total_qty = 0
        for item in items:
            qty = item.get("qty", 0)
            try:
                total_qty += int(qty)
            except (ValueError, TypeError):
                continue  # Skip non-numeric qty

        df.at[i, "ticket_qty"] = total_qty

     except Exception as e:
        st.warning(f"Couldn't fetch tickets for booking {row['booking_id']}: {e}")


    
    # Add a column for ticket quantity (default to 0)
    df["ticket_qty"] = 0

# Loop through each booking to get detailed ticket count (safe handling)
    for i, row in df.iterrows():
     try:
        details = fetch_booking_details(row["booking_id"])
        booking_info = details.get("booking", {})
        items = booking_info.get("items", [])

        # Normalize items to a list of dicts
        if isinstance(items, dict):
            items = list(items.values())
        elif isinstance(items, str) or not isinstance(items, list):
            items = []  # Unexpected structure, skip this entry

        total_qty = 0
        for item in items:
          qty = item.get("qty", 0)
        try:
         total_qty += int(qty)
        except (ValueError, TypeError):
         continue  # skip if not a number


        df.at[i, "ticket_qty"] = total_qty

     except Exception as e:
        st.warning(f"Couldn't fetch tickets for booking {row['booking_id']}: {e}")



    if df.empty:
        st.warning("No bookings found.")
        st.stop()

    df["created_date"] = pd.to_datetime(df["created_date"], unit="s")
    df["total"] = pd.to_numeric(df["total"], errors="coerce")
    df["paid_total"] = pd.to_numeric(df["paid_total"], errors="coerce")
    df["status_name"] = df["status_name"].fillna("Unknown")
    df["month"] = df["created_date"].dt.to_period("M")
    df["day"] = df["created_date"].dt.day_name()
    df["hour"] = df["created_date"].dt.hour
    df["summary"] = df["summary"].astype(str).str.strip()

    if status_filter != "All":
        df = df[df["status_name"] == status_filter]

    if search:
        df = df[df.apply(lambda row: search in str(row.get("customer_name", "")).lower()
                                   or search in str(row.get("customer_email", "")).lower()
                                   or search in str(row.get("code", "")).lower(), axis=1)]

    total_bookings = len(df)
    total_revenue = df["total"].sum()
    avg_booking = df["total"].mean()
    paid_pct = (df["paid_total"] > 0).mean() * 100
    repeat_rate = df["customer_email"].duplicated().mean() * 100

    kpi_data = {
        "Total Bookings": total_bookings,
        "Total Revenue": f"£{total_revenue:,.2f}",
        "Avg per Booking": f"£{avg_booking:,.2f}",
        "Paid %": f"{paid_pct:.1f}%",
        "Repeat Customers %": f"{repeat_rate:.1f}%"
    }

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Bookings", total_bookings)
    k2.metric("Total Revenue", f"£{total_revenue:,.2f}")
    k3.metric("Avg per Booking", f"£{avg_booking:,.2f}")
    k4.metric("Paid %", f"{paid_pct:.1f}%")
    k5.metric("Repeat Cust %", f"{repeat_rate:.1f}%")

    # PDF Download
    top_tour = df.groupby("summary")["total"].sum().idxmax()
    top_day = df["day"].mode()[0]
    recent_rows = df.sort_values("created_date", ascending=False).head(5).to_dict(orient="records")
    date_range = f"{start.strftime('%d %b %Y')} to {end.strftime('%d %b %Y')}"
    pdf_bytes = create_detailed_pdf_summary(kpi_data, date_range, top_tour, top_day, recent_rows)

        # Generate today's date string for filename
    today_str = datetime.today().strftime("%Y-%m-%d")
    pdf_filename = f"shoebox_summary_{today_str}.pdf"

    # Sidebar PDF download button
    st.sidebar.markdown("###  Download Report")
    st.sidebar.download_button(
        label="⬇️ Download PDF",
        data=pdf_bytes,
        file_name=pdf_filename,
        mime="application/pdf"
    )


        # --- Charts Section ---
    st.markdown("### 📈 Insights")

    # First row
    col1, col2 = st.columns(2)

    with col1:
        time_series = df.groupby(df["created_date"].dt.date).size().reset_index(name="Bookings")
        fig1 = px.line(time_series, x="created_date", y="Bookings", title=" Bookings Over Time")
        st.plotly_chart(fig1, use_container_width=True, key="chart_bookings_time")
        st.caption("This line chart shows the total number of bookings made per day over the selected date range.")

    with col2:
        pie = df["status_name"].value_counts().reset_index()
        pie.columns = ["Status", "Count"]
        fig2 = px.pie(pie, names="Status", values="Count", title=" Booking Status Breakdown")
        st.plotly_chart(fig2, use_container_width=True, key="chart_status_pie")
        st.caption("This pie chart shows the proportion of bookings by their current status (e.g., Paid, Web-Pre-Booking, Waiting, Deposit).")

    # Second row
    col3, col4 = st.columns(2)

    with col3:
        tour_rev = df.groupby("summary")["total"].sum().reset_index().sort_values("total", ascending=False)
        fig3 = px.bar(tour_rev, x="summary", y="total", title=" Revenue by Tour", text_auto=True)
        st.plotly_chart(fig3, use_container_width=True, key="chart_revenue_tour")
        st.caption("This bar chart breaks down total revenue generated per tour during the selected date range.")

    with col4:
    # Ensure month column
     df["month"] = df["created_date"].dt.to_period("M")

    # Generate all months between selected range
    all_months = pd.period_range(start=start, end=end, freq='M')

    # Aggregate revenue per month
    monthly_revenue = df.groupby(df["created_date"].dt.to_period("M"))["total"].sum()

    # Reindex to ensure missing months are shown with 0
    monthly_revenue = monthly_revenue.reindex(all_months, fill_value=0).reset_index()
    monthly_revenue.columns = ["Month", "Total Revenue"]

    # Format month names for display (e.g., "Jul 2025")
    monthly_revenue["Month"] = monthly_revenue["Month"].dt.strftime("%b %Y")

    # Plot
    fig_monthly = px.bar(
        monthly_revenue,
        x="Month",
        y="Total Revenue",
        title="📆 Total Monthly Revenue",
        text_auto=True
    )

    st.plotly_chart(fig_monthly, use_container_width=True)
    st.caption("This bar chart shows the total revenue collected for each calendar month in the selected date range, including months with no bookings.")


    c5, c6 = st.columns(2)
    with c5:
     day_counts = df["day"].value_counts().reset_index()
    day_counts.columns = ["Day", "Count"]
    st.plotly_chart(
        px.bar(day_counts, x="Day", y="Count", title=" Bookings by Day"),
        use_container_width=True,
        key="chart_by_day"
    )
    with c6:
     hour_counts = df["hour"].value_counts().sort_index().reset_index()
    hour_counts.columns = ["Hour", "Count"]
    st.plotly_chart(
        px.bar(hour_counts, x="Hour", y="Count", title=" Bookings by Hour"),
        use_container_width=True,
        key="chart_by_hour"
    )

# NEW COMPARISON CHARTS
    st.markdown("###  Extended Time-Based Comparisons")
    df["week"] = df["created_date"].dt.to_period("W").apply(lambda r: r.start_time)
    df["year"] = df["created_date"].dt.year
    df["week_num"] = df["created_date"].dt.isocalendar().week
    df["quarter"] = df["created_date"].dt.to_period("Q").astype(str)

    st.subheader(" Weekly Breakdown by Product/Item")
    weekly_breakdown = df.groupby(["week", "summary"])["total"].sum().reset_index()
    st.plotly_chart(
    px.bar(
        weekly_breakdown,
        x="week",
        y="total",
        color="summary",
        title="Weekly Revenue by Product/Item",
        barmode="group"
    ),
    use_container_width=True,
    key="chart_weekly_breakdown"
)

    st.subheader(" Weekly Revenue Comparison (Last 3 Years)")
    weekly_compare = df.groupby(["year", "week_num"])["total"].sum().reset_index()
    st.plotly_chart(
    px.line(
        weekly_compare,
        x="week_num",
        y="total",
        color="year",
        markers=True,
        title="Weekly Revenue Trends by Year"
    ),
    use_container_width=True,
    key="chart_weekly_trend"
)

    st.subheader(" Quarterly Revenue Comparison (Last 3 Years)")
    quarterly_compare = df.groupby(["year", "quarter"])["total"].sum().reset_index()
    st.plotly_chart(
    px.bar(
        quarterly_compare,
        x="quarter",
        y="total",
        color="year",
        barmode="group",
        title="Quarterly Revenue Trends by Year"
    ),
    use_container_width=True,
    key="chart_quarterly_trend"
)

    
   # === STOCK AVAILABILITY & MISSED REVENUE ===
    st.markdown("##  Stock Availability & Missed Revenue")
    st.caption("Filtered to tours only. Booked ticket totals are now based on number of tickets sold, not just bookings.")

    stock_start = st.date_input("Start Date for Stock Analysis", value=date.today())
    stock_end = st.date_input("End Date for Stock Analysis", value=date.today() + timedelta(days=30))

    num_days = (stock_end - stock_start).days + 1


    try:
     df["summary"] = df["summary"].astype(str).str.strip()
     df["event_date"] = pd.to_datetime(df["date_desc"], errors="coerce")
    
    # Filter date range for stock analysis
     df_stock = df[(df["event_date"] >= pd.Timestamp(stock_start)) & (df["event_date"] <= pd.Timestamp(stock_end))].copy()
    
    # Exclude meeting room entries
     df_stock = df_stock[~df_stock["summary"].str.contains("Meeting Room", case=False, na=False)]

    # Dynamic list of all valid tour names
     all_products = sorted(df_stock["summary"].dropna().unique())
     


    # Estimated prices for each tour (can add more if needed)
     product_prices = {
        "Norwich's Hidden Street Tour": 15.00,
        "The Chronicles of Christmas": 10.00,
        "The TIPSY Tavern Trail Tour": 14.00,
        "The Tavern Trail Tour": 12.00,
        "The Norwich Knowledge Tour": 12.00,
        "City of Centuries Tour": 13.00,
        "The Matriarchs, Mayors & Merchants Tour": 14.00,
        "Secrets of the Tunnels - Thrilling underground escape game": 16.00,
        "Magnificent Marble Hall": 14.00  
    }

     stock_rows = []

     for product in all_products:
      product_df = df_stock[df_stock["summary"] == product]
      tickets_booked = product_df["ticket_qty"].fillna(0).sum()

      daily_capacity = 12  # default per tour per day
      total_capacity = daily_capacity * num_days  # make sure num_days is defined

      available = total_capacity - tickets_booked
      avg_price = product_prices.get(product, 0)
      lost_revenue = available * avg_price

      stock_rows.append({
        "Product": product,
        "Booked Tickets": tickets_booked,
        "Capacity": total_capacity,
        "Available": available,
        "Price (£)": avg_price,
        "Potential Revenue Lost (£)": round(lost_revenue, 2)
    })



     stock_df = pd.DataFrame(stock_rows)

     with st.expander("📋 Full Stock & Revenue Table"):
        st.dataframe(stock_df)
        
     stock_df["Product"] = stock_df["Product"].astype(str).str.strip()
   

     st.plotly_chart(
        px.bar(
            stock_df,
            x="Product",
            y=["Booked Tickets", "Available"],
            barmode="stack",
            title="Stock vs Tickets Booked"
        ),
        use_container_width=True
    )

     st.plotly_chart(
        px.bar(
            stock_df,
            x="Product",
            y="Potential Revenue Lost (£)",
            title="Potential Revenue Lost",
            text_auto=True
        ),
        use_container_width=True
    )

    except Exception as e:
     st.warning("⚠️ Error calculating stock and lost revenue.")
     st.error(f"Error: {e}")


 # === PRODUCT FILTERING ===
    st.markdown("###  Product & Category Filters (Optional View)")

    # Product filter
    product_options = ["All"] + sorted(df["summary"].dropna().unique())
    selected_product = st.selectbox(" Filter by Product", product_options, index=0)

    if selected_product != "All":
        df = df[df["summary"] == selected_product]

    # Optional: Category filter if available
    if "category" in df.columns:
        category_options = ["All"] + sorted(df["category"].dropna().unique())
        selected_category = st.selectbox(" Filter by Category", category_options, index=0)

        if selected_category != "All":
            df = df[df["category"] == selected_category]

        st.info(f"Showing data for category: **{selected_category}**")

    if selected_product != "All":
        st.info(f"Showing data for product: **{selected_product}**")

except Exception as e:
    st.error(f"❌ Error loading dashboard: {e}")

