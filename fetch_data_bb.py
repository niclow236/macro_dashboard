import os
import pandas as pd
from xbbg import blp
import tkinter as tk
from tkinter import filedialog, messagebox
import warnings
warnings.filterwarnings("ignore")

# --- GUI FUNCTIONS ---
def show_messagebox(title, message):
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes('-topmost', True)
    messagebox.showinfo(title, message, parent=root)
    root.destroy()

def prompt_save_file(title="Save Excel File"):
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes('-topmost', True)
    file_path = filedialog.asksaveasfilename(
        title=title,
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")],
        initialfile="macro_data_bb.xlsx"
    )
    root.destroy()
    return file_path

# --- CONFIGURATION ---
start_date = '2018-01-01'

# --- TICKERS ---
us_tickers = {
    'GDP_YoY': 'GDP CYOY Index',
    'IP_YoY': 'IP YOY Index',
    'Unemployment': 'USURTOT Index',
    'Core_CPI_YoY': 'CPURNSA Index',
    'PPI_YoY': 'FDIUFDYO Index',
    'Manufacturing_PMI': 'NAPMPMI Index',
    '10Y': 'GT10 Govt',
    '2Y': 'GT2 Govt', 
    'HY_OAS': 'LF98OAS Index',
    'Retail_Sales_MoM': 'RSTAMOM Index',
    'Capacity_Utilization': 'CPTICHNG Index',
    'Core_PCE_YoY': 'PCE CMOM Index',
    'Services_PMI': 'NAPMNMI Index',
    'SmallBiz_Sentiment': 'SBOITOTL Index',
    'Home_Sales': 'ETSLTOTL Index',
    'Jobless_Claims_4WMA': 'INJCJC Index'
}

cn_tickers = {
    'GDP_YoY': 'CNGDPC$Y Index',
    'Core_CPI_YoY': 'CNCPIYOY Index',
    'Unemployment': 'CNUESRU Index',
    'Manufacturing_PMI': 'CPMINDX Index',
    'Industrial_Value_Added_YoY': 'CHVAIOY Index',
    'Retail_Sales_YoY': 'CNRSCYOY Index',
    'Service_Production_Index': 'CNSF Index',
    'PPI_YoY': 'CHEFTYOY Index',
    'Exports_YoY': 'CNFREXPY Index',
    'Money_Supply_M2_YoY': 'CNMS2YOY Index',
    'SHIBOR_1M': 'SHIF1M Index',
    'Real_Estate_Climate_Index_YoY': 'CHRXCINY Index',
}

jp_tickers = {
    'GDP_YoY': 'JGDPAGDP Index',
    'Unemployment': 'JNUNRT Index',
    'Core_CPI_YoY': 'JNCPIYOY Index',
    'PPI_YoY': 'JNWSDYOY Index',
    'Retail_Sales_YoY': 'JNNETYOY Index',
    'Exports_YoY': 'JNTBEXPY Index',
    '10Y': 'GJGB10 Index',
    '2Y': 'GJGB2 Index',
    'Money_Supply_M2_YoY': 'JMNSM2Y Index',
    'Tankan_Business_Conditions_LE_Mfg': 'JNTSMFG Index',
    'Industrial_Production_YoY': 'JNIPYOY Index',
    'Consumer_Confidence': 'JCOMACF Index',
    'Business_Confidence_All_Industry': 'JSMEALLI Index',
}

eu_tickers = {
    'GDP_YoY': 'EUGNEMUY Index',
    'Unemployment': 'UMRTEMU Index',
    'Core_CPI_YoY': 'CPEXEMUY Index',
    'Composite_PMI': 'MPMIEZCA Index',
    'Money_Supply_M3_YoY': 'ECMAM3YY Index',
    'Capacity_Utilization': 'EUUCEMU Index',
    'Consumer_Confidence': 'EUCCEMU Index',
    'Yield_Spread': 'EUCBEIOR Index',
    'Credit_Impulse': 'BCMPCIGD Index',
    'IP_YoY': 'EUIPEMUY Index',
    'Retail_Expectations': 'EUR4EMU Index',
    'Economic_Sentiment': 'EUESEMU Index'
}

sk_tickers = {
    'GDP_YoY': 'KOGDPYOY Index',
    'Unemployment': 'KOEAUERS Index',
    'Core_CPI_YoY': 'KOCPIYOY Index',
    'PPI_YoY': 'KOPPIYOY Index',
    'Retail_Sales_YoY': 'KORSTY Index',
    'Exports_YoY': 'KOEXTOTY Index',
    'Manufacturing_PMI': 'MPMIKRMA Index',
    '10Y': 'GTKRW10Y Govt',
    '2Y': 'GTKRW2Y Govt',
    'IP_YoY': 'KOIPIY Index',
    'Business_Sentiment': 'KOBSCBSI Index',
    'Household_Debt': 'KOHHD Index',
}

au_tickers = {
    'GDP_YoY': 'AUNAGDPY Index',
    'Unemployment': 'AULFUNEM Index',
    'Core_CPI_YoY': 'ACPMTRNY Index',
    'PPI_YoY': 'AUPPFYOY Index',
    'Retail_Sales_YoY': 'AURSTYSA Index',
    'Exports_YoY': 'AUITEXGY Index',
    'Composite_PMI': 'MPMIAUCA Index',
    '10Y': 'GTAUD10Y Govt',
    '2Y': 'GTAUD2Y Govt',
    'Mining_Labor_YoY': 'AULQMINY Index',
    'Money_Supply_M3_YoY': 'AUM3Y Index',
    'Housing_Loan_Interest_Rate': 'AILRHLBS Index',
    'Business_Conditions': 'NABSCOND Index',
    'Business_Confidence': 'NABSCONF Index',
}


# --- DATA FETCH FUNCTION ---
def fetch_macro_data(ticker_dict, label):
    frames = []
    for name, ticker in ticker_dict.items():
        try:
            df = blp.bdh(ticker, 'PX_LAST', start_date=start_date).dropna()
            df.columns = [name]
            df.index = pd.to_datetime(df.index)
            frames.append(df)
            print(f"✅ Fetched {label}: {name}")
        except Exception as e:
            print(f"⚠️ Failed {label} - {name}: {e}")
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()

# --- MAIN ---
if __name__ == "__main__":
    show_messagebox("Bloomberg Fetch", "Fetching macroeconomic data from Bloomberg.\nClick OK to select save location.")
    save_path = prompt_save_file("Save Macroeconomic Data As")

    if not save_path:
        print("❌ Save location not selected. Exiting.")
        exit()

    us_df = fetch_macro_data(us_tickers, "USA")
    cn_df = fetch_macro_data(cn_tickers, "China")
    jp_df = fetch_macro_data(jp_tickers, "Japan")
    eu_df = fetch_macro_data(eu_tickers, "Eurozone")
    sk_df = fetch_macro_data(sk_tickers, "South Korea")
    au_df = fetch_macro_data(au_tickers, "Australia")

    if not us_df.empty or not cn_df.empty:
        with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
            if not us_df.empty:
                us_df.to_excel(writer, sheet_name='USA')
            if not cn_df.empty:
                cn_df.to_excel(writer, sheet_name='China')
            if not jp_df.empty:
                jp_df.to_excel(writer, sheet_name='Japan')
            if not eu_df.empty:
                eu_df.to_excel(writer, sheet_name='Eurozone')
            if not sk_df.empty:
                sk_df.to_excel(writer, sheet_name='South Korea')
            if not au_df.empty:
                au_df.to_excel(writer, sheet_name='Australia')
        show_messagebox("Success", f"Data saved to:\n{save_path}")
        print(f"✅ Exported data to {save_path}")
    else:
        show_messagebox("Error", "No data was fetched. Please check your Bloomberg connection.")
        print("❌ No data fetched.")
