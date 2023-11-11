
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import requests
from bs4 import BeautifulSoup
import urllib
import streamlit_antd_components as sac

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide", page_title='yFinance Dashboard')

#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret
#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

# SideBar START----------------------------------------------------------------

st.sidebar.header(':violet[_Yahoo_] Finance Dashboard')
st.sidebar.caption('All rights reserved to Yahoo')

ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

# Assigning company symbol to ticker and making it globally accessible---------
global ticker
ticker = st.sidebar.selectbox("Company Symbol", ticker_list)
def GetCompanyInfo(ticker):
    return YFinance(ticker).info
info = GetCompanyInfo(ticker)

with st.sidebar:
    sac.buttons([
        sac.ButtonsItem(label='Update', icon='arrow-clockwise', color='#6f42c1'),
        ], key='refreshButton', type='primary')

if st.session_state['refreshButton']:
 def refresh_stock_data():
    updated_data = yf.Ticker(ticker).history(period='1d')
    return updated_data
    st.success("Stock data refreshed successfully!")
    if st.sidebar.button("Update"):
        refresh_stock_data()
        
# SideBar END------------------------------------------------------------------

cName = st.header(info["website"], divider = "rainbow")
st.caption('Compiled by Srivardhan SriHari')

# Defining the main nav bar----------------------------------------------------
sac.segmented(
items=[
    sac.SegmentedItem(label='Summary', icon='house'),
    sac.SegmentedItem(label='Chart', icon='bar-chart'),
    sac.SegmentedItem(label='Financials', icon='cash-coin'),
    sac.SegmentedItem(label='Monte Carlo Simulation', icon='diagram-3'),
    sac.SegmentedItem(label='Updates', icon='newspaper')
    ], color='violet', key='homeTab', align='center'
)
# Tab1 START===================================================================

if st.session_state['homeTab'] == 'Summary':
    with st.container():
        col1, col2 = st.columns([2,2])
    
        with col1:
            col1a, col1b = st.columns(2)
            with col1a:
                prevCloseVar = str(info["previousClose"])
                openVar = str(info["open"])
                bidVar = str(info["bid"])
                askVar = str(info["ask"])
                days_range = str(info["dayLow"]) + '-' + str(info["dayHigh"])
                weekRangeVar = str(info["fiftyTwoWeekLow"]) + '-' + str(info["fiftyTwoWeekHigh"])
                volVar = str(info["volume"])
                avgVolVar = str(info["averageVolume"])
                
                valueArr = {
                            prevCloseVar :'Previous Close',
                            openVar        :'Open',
                            bidVar         :'Bid',
                            askVar          :'Ask',
                            days_range  : 'Day\'s Range',
                            weekRangeVar : '52 Week Range',
                            volVar       :'Volume',
                            avgVolVar :'Average Volume'
                            }
    
                swapped_dict = {v: k for k, v in valueArr.items()} 
                swapped_dict = pd.DataFrame({"":pd.Series(swapped_dict)})  # Convert to DataFrame
                st.dataframe(swapped_dict, use_container_width=True)
            
            with col1b:
    
                price = info["currentPrice"]
                EPS = info["trailingEps"]
                PE = round(price / EPS, 2)
                
# Fetching certain DataPoints using HTML since they are unavailable in the .info api------
#---------------Define the URL for the Yahoo Finance page
                url = f'https://finance.yahoo.com/quote/{ticker}'
                page = requests.get(url)
                soup = BeautifulSoup(page.content, "html.parser")
                
#---------------Earnings Date-------------------------------------------
                results = soup.find(attrs={"data-test" : "EARNINGS_DATE-value"})
                output = results.contents
                print(str(output))
                original_string = str(output)
                unwanted_characters = "[]''/,<>"
                words_to_remove = ["span"]
                for word in words_to_remove:
                    original_string = original_string.replace(word, "")
                modified_string = original_string
                for char in unwanted_characters:
                    modified_string = modified_string.replace(char, "")
                print(modified_string)
#---------------Earnings Date---------------------------------------------     
                
               
#---------------Forward dividend and yield------------------------------
                results2 = soup.find(attrs={"data-test" : "DIVIDEND_AND_YIELD-value"})
                output2 = results2.contents
                print(str(output2))
                original_string2 = str(output2)
                unwanted_characters2 = "[]''/,<>"
                words_to_remove2 = ["span"]
                for word in words_to_remove2:
                    original_string2 = original_string2.replace(word, "")
                modified_string2 = original_string2
                for char in unwanted_characters2:
                    modified_string2 = modified_string2.replace(char, "")
                print(modified_string2)
#---------------Forward dividend and yield--------------------------------     
                
#---------------1y Target Est-------------------------------------------
                results3 = soup.find(attrs={"data-test" : "ONE_YEAR_TARGET_PRICE-value"})
                output3 = results3.contents
                print(str(output3))
                original_string3 = str(output3)
                unwanted_characters3 = "[]''/,<>"
                words_to_remove3 = ["span"]
                for word in words_to_remove3:
                    original_string3 = original_string3.replace(word, "")
                modified_string3 = original_string3
                for char in unwanted_characters3:
                    modified_string3 = modified_string3.replace(char, "")
                print(modified_string3)
#---------------1y Target Est---------------------------------------------    
                
#---------------Ex-Dividend Date----------------------------------------
                results4 = soup.find(attrs={"data-test" : "EX_DIVIDEND_DATE-value"})
                output4 = results4.contents
                print(str(output4))
                original_string4 = str(output4)
                unwanted_characters4 = "[]''/,<>"
                words_to_remove4 = ["span"]
                for word in words_to_remove4:
                    original_string4 = original_string4.replace(word, "")
                modified_string4 = original_string4
                for char in unwanted_characters4:
                    modified_string4 = modified_string4.replace(char, "")
                print(modified_string4)
#---------------Ex-Dividend Date------------------------------------------    
                
                marketCapVar = str(info["marketCap"])
                betaVar = str(info["beta"])
                peVar = str(PE)
                epsVar = str(info["trailingEps"])
                earningsVar = str(modified_string)
                forwardVar = str(modified_string2)
                exDviVar = str(modified_string4)
                targetEstVar = str(modified_string3)
                valueArr = {
                            marketCapVar :'Market Cap',
                            betaVar        :'Beta 5Y Monthly',
                            peVar         :'PE Ratio (TTM)',
                            epsVar          :'EPS (TTM)',
                            earningsVar  : 'Earnings Date',
                            forwardVar : 'Forward Dividend & Yield',
                            exDviVar       :'Ex-Dividend Date',
                            targetEstVar :'1y Target Est'
                            }
                swapped_dict = {v: k for k, v in valueArr.items()} 
#---------------Convert to DataFrame-------------------------------------------------
                swapped_dict = pd.DataFrame({"":pd.Series(swapped_dict)})  
                st.dataframe(swapped_dict, use_container_width=True)

        with col2:
            global chartSel
            start_date = datetime.today().date() - timedelta(days=30)
            with st.container():
                colr1, colr2 = st.columns([4,1])
                with colr1:
                    sac.buttons([
                    sac.ButtonsItem(label='1M', color='#6f42c1'),
                    sac.ButtonsItem(label='3M', color='#6f42c1'),
                    sac.ButtonsItem(label='6M', color='#6f42c1'),
                    sac.ButtonsItem(label='YTD', color='#6f42c1'),
                    sac.ButtonsItem(label='1Y', color='#6f42c1'),
                    sac.ButtonsItem(label='3Y', color='#6f42c1'),
                    sac.ButtonsItem(label='5Y', color='#6f42c1'),
                    sac.ButtonsItem(label='Max', color='#6f42c1'),
                    ], format_func='title', align='start', type='link', key = 'chartChange', index = 2)
                    
#-------------------Determine the start and end dates based on the selected time range
                    if st.session_state["chartChange"] == "1M":
                        start_date = datetime.today().date() - timedelta(days=30)
                    elif st.session_state["chartChange"] == "3M":
                        start_date = datetime.today().date() - timedelta(days=90)
                    elif st.session_state["chartChange"] == "6M":
                        start_date = datetime.today().date() - timedelta(days=180)
                    elif st.session_state["chartChange"] == "YTD":
                        start_date = datetime(datetime.today().year, 1, 1).date()
                    elif st.session_state["chartChange"] == "1Y":
                        start_date = datetime.today().date() - timedelta(days=365)
                    elif st.session_state["chartChange"] == "3Y":
                        start_date = datetime.today().date() - timedelta(days=1095)
                    elif st.session_state["chartChange"] == "5Y":
                        start_date = datetime.today().date() - timedelta(days=1825)
                    elif st.session_state["chartChange"] == "Max":
                        if 'stock_price' in locals() or 'stock_price' in globals():
                            start_date = info['Date'].min().date()
                        else:
                            start_date = datetime.today().date() - timedelta(days=30)
                with colr2:
                    chartSel = st.selectbox("Chart Type", ("Line", "CandleStick", "Area"), index = 2)
 
            end_date = datetime.today().date()
            
            def GetStockData(ticker, start_date, end_date):
                stock_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
#---------------Drop the indexes
                stock_df.reset_index(inplace=True)
#---------------Convert date-time to date
                stock_df['Date'] = stock_df['Date'].dt.date 
                return stock_df; 
            
            show_data_tab1 = st.checkbox("Show data table")
            
            if ticker != '':
                stock_price = GetStockData(ticker, start_date, end_date)
                if ticker != '' and chartSel == 'Line':     
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=stock_price['Date'], y=stock_price['Close'], mode='lines', name='Stock Price'))
                    fig.update_layout(xaxis_title='Date', yaxis_title='Close Price', height=300)
                    st.plotly_chart(fig, use_container_width=True)
                elif ticker != '' and chartSel == 'CandleStick':
                    fig = go.Figure(data=[go.Candlestick(x=stock_price['Date'],
                                             open=stock_price['Open'],
                                             high=stock_price['High'],
                                             low=stock_price['Low'],
                                             close=stock_price['Close'],
                                             name='Stock Price')])
                    fig.update_layout(xaxis_title='Date', yaxis_title='Price', title='', height=300)
                    st.plotly_chart(fig, use_container_width=True)
                elif ticker != '' and chartSel == 'Area':
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=stock_price['Date'], y=stock_price['Close'], fill='tozeroy', mode='lines', name='Stock Price', line=dict(color='rgba(111, 66, 193, 0.2)')))
                    fig.update_layout(xaxis_title='Date', yaxis_title='Close Price', title='Area Chart for Stock', height=300)
                    st.plotly_chart(fig, use_container_width=True)
                if show_data_tab1:
                    st.dataframe(stock_price, hide_index=True, use_container_width=True)
                fig.add_trace(go.Bar(x=stock_price['Date'], y=stock_price['Volume'], name='Volume', marker=dict(color='rgba(255, 255, 255, 0.3)')))
                fig.update_layout(yaxis2=dict(title='Volume', overlaying='y', side='right'))
                
    st.divider()
    
    cProfile, cDesc, cStakeholders = st.columns([2,3.5,1])
    with cProfile:
        st.subheader('Profile', divider='rainbow')
        officeDf = pd.DataFrame(info["companyOfficers"])
        st.dataframe(officeDf[['name', 'title']], hide_index = True)
    with cDesc:
        st.subheader('Description', divider='rainbow')
        brief = str(info["longBusinessSummary"])
        st.caption(brief)
        
# Tab1 END===================================================================

# Tab2 START===================================================================
if st.session_state['homeTab'] == 'Chart':
            global chartVar
            start_date = datetime.today().date()
            with st.container():
                colr1, colr2 = st.columns([4,0.5])
                with colr1:
                    sac.buttons([
                    sac.ButtonsItem(label='Date Range', icon='calendar-range', color='#6f42c1'),
                    sac.ButtonsItem(label='1M', color='#6f42c1'),
                    sac.ButtonsItem(label='3M', color='#6f42c1'),
                    sac.ButtonsItem(label='6M', color='#6f42c1'),
                    sac.ButtonsItem(label='YTD', color='#6f42c1'),
                    sac.ButtonsItem(label='1Y', color='#6f42c1'),
                    sac.ButtonsItem(label='3Y', color='#6f42c1'),
                    sac.ButtonsItem(label='5Y', color='#6f42c1'),
                    sac.ButtonsItem(label='Max', color='#6f42c1'),
                    ], format_func='title', align='start', type='link', key = 'chartChange', index = 0)
                    
                    if st.session_state["chartChange"] == "Date Range":
                        col1, col2 = st.columns([1,4])
                        with col1:
                            date_range = st.date_input(label = "Date Range", value = None, key = 'dateRange', format = "YYYY-MM-DD")
                        if st.session_state.dateRange == None:
                            start_date = start_date = datetime.today().date() - timedelta(days=30)
                        elif st.session_state.dateRange != None:
                            start_date = date_range
                    elif st.session_state["chartChange"] == "1M":
                        start_date = datetime.today().date() - timedelta(days=30)
                    elif st.session_state["chartChange"] == "3M":
                        start_date = datetime.today().date() - timedelta(days=90)
                    elif st.session_state["chartChange"] == "6M":
                        start_date = datetime.today().date() - timedelta(days=180)
                    elif st.session_state["chartChange"] == "YTD":
                        start_date = datetime(datetime.today().year, 1, 1).date()
                    elif st.session_state["chartChange"] == "1Y":
                        start_date = datetime.today().date() - timedelta(days=365)
                    elif st.session_state["chartChange"] == "3Y":
                        start_date = datetime.today().date() - timedelta(days=1095)
                    elif st.session_state["chartChange"] == "5Y":
                        start_date = datetime.today().date() - timedelta(days=1825)
                    elif st.session_state["chartChange"] == "Max":
                        if 'stock_price' in locals() or 'stock_price' in globals():
                            start_date = info['Date'].min().date()
                        else:
                            start_date = datetime.today().date() - timedelta(days=30)

                with colr2:
                    chartVar = st.selectbox("Chart Type", ("Line", "CandleStick"))
            
            endDate = datetime.today().date()
                
            def GetStockData(ticker, start_date, endDate):
                stock_df = yf.Ticker(ticker).history(start = start_date, end = endDate)
#---------------Drop the indexes
                stock_df.reset_index(inplace=True)
#---------------Convert date-time to date
                stock_df['Date'] = stock_df['Date'].dt.date 
                return stock_df;
#---------------Calculating simple moving average           
            def calculate_sma(data, window_size):
                return data['Close'].rolling(window=window_size).mean() 
            
            if ticker != '':
                stock_price = GetStockData(ticker, start_date, endDate)
                sma_window_size = 50
                stock_price['SMA'] = calculate_sma(stock_price, sma_window_size)
                if ticker != '' and chartVar == 'Line':     
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=stock_price['Date'], y=stock_price['Close'], mode='lines', name='Stock Price', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=stock_price['Date'], y=stock_price['SMA'], mode='lines', name=f'SMA ({sma_window_size} days)', line=dict(color='orange')))
                    fig.update_layout(xaxis_title='Date', yaxis_title='Close Price', template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif ticker != '' and chartVar == 'CandleStick':
                    fig = go.Figure(data=[go.Candlestick(x=stock_price['Date'],
                                              open=stock_price['Open'],
                                              high=stock_price['High'],
                                              low=stock_price['Low'],
                                              close=stock_price['Close'],
                                              name='Stock Price',
                                              increasing_line_color='green', decreasing_line_color='red')])
                    fig.add_trace(go.Scatter(x=stock_price['Date'], y=stock_price['SMA'], mode='lines', name=f'SMA ({sma_window_size} days)', line=dict(color='orange')))
                    fig.update_layout(xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
                    
                fig.add_trace(go.Bar(x=stock_price['Date'], y=stock_price['Volume'], name='Volume', marker=dict(color='rgba(255, 255, 255, 0.3)')))
                fig.update_layout(yaxis2=dict(title='Volume', overlaying='y', side='right'))

# Tab2 END=================================================================

# Tab3 START=================================================================
if st.session_state['homeTab'] == 'Financials':
    col1, col2 = st.columns([2, 2])
    with col1:
        sac.segmented(
        items=[
            sac.SegmentedItem(label='Income Statement'),
            sac.SegmentedItem(label='Balance Sheet'),
            sac.SegmentedItem(label='Cash Flow'),
            ], format_func='title', color='violet', bg_color='transparent', key = "mainTab"
        )
    with col2:
        sac.segmented(
        items=[
            sac.SegmentedItem(label='Annual', icon='calendar3'),
            sac.SegmentedItem(label='Quarterly', icon='calendar4-week'),
            ], format_func='title', color='gray', align='end', bg_color='transparent', key = "subTab"
        )
    st.subheader(st.session_state['mainTab']) 
    st.caption('**All numbers in thousands**')
        
    if st.session_state["mainTab"] == "Income Statement" and st.session_state["subTab"] == "Annual":
        annIncomeStat = yf.Ticker(ticker).income_stmt
        st.dataframe(annIncomeStat, use_container_width=True)
    elif st.session_state["mainTab"] == "Income Statement" and st.session_state["subTab"] == "Quarterly":
        quartIncomeStat = yf.Ticker(ticker).quarterly_income_stmt
        st.dataframe(quartIncomeStat, use_container_width=True)
    elif st.session_state["mainTab"] == "Balance Sheet" and st.session_state["subTab"] == "Quarterly":
        quartBalSheet = yf.Ticker(ticker).quarterly_balance_sheet
        st.dataframe(quartBalSheet, use_container_width=True)
    elif st.session_state["mainTab"] == "Balance Sheet" and st.session_state["subTab"] == "Annual":
        annBalSheet = yf.Ticker(ticker).balance_sheet
        st.dataframe(annBalSheet, use_container_width=True)
    elif st.session_state["mainTab"] == "Cash Flow" and st.session_state["subTab"] == "Quarterly":
        annCashFlow = yf.Ticker(ticker).cashflow
        st.dataframe(annCashFlow, use_container_width=True)
    elif st.session_state["mainTab"] == "Cash Flow" and st.session_state["subTab"] == "Annual":
        quartCashFlow = yf.Ticker(ticker).quarterly_cashflow
        st.dataframe(quartCashFlow, use_container_width=True)
        
# Tab3 END==================================================================


# Tab4 START=================================================================
if st.session_state['homeTab'] == 'Monte Carlo Simulation':
    def monte_carlo_simulation(ticker, n_simulations, time_horizon):
# Get historical stock data
        stock_data = yf.download(ticker, start=(pd.to_datetime('today') - pd.DateOffset(days=time_horizon)).date())
        returns = stock_data['Close'].pct_change().dropna()
    
# Calculate daily log returns mean and standard deviation
        mu = returns.mean()
        sigma = returns.std()
    
# Generate simulated daily returns
        daily_returns = np.random.normal(mu, sigma, size=(time_horizon, n_simulations))
    
# Calculate cumulative returns
        cumulative_returns = np.exp(np.cumsum(daily_returns, axis=0))
    
# Get the last closing price
        last_close_price = stock_data['Close'].iloc[-1]
    
# Calculate simulated stock prices
        simulated_prices = last_close_price * cumulative_returns
    
        return simulated_prices

# Function to calculate Value at Risk (VaR)
    def calculate_var(simulated_prices, confidence_level=0.95):
        percentiles = np.percentile(simulated_prices, [100 * (1 - confidence_level)], axis=1)
        return percentiles

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        n_simulations = st.selectbox('Number of Simulations:', [200, 500, 1000])
    with col2:
        time_horizon = st.selectbox('Time Horizon (Days):', [30, 60, 90])
    
# Run simulation and calculate VaR
    simulated_prices = monte_carlo_simulation(ticker, n_simulations, time_horizon)
    var_95 = calculate_var(simulated_prices)
    
    st.subheader('Monte Carlo Simulation - Stock Closing Price')
# Plot simulation results
    plt.figure(figsize=(11, 2))
    plt.plot(simulated_prices)
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    st.pyplot()
    
# Display VaR at 95% confidence level
    st.subheader('Value at Risk (VaR) at 95% confidence level:', divider='rainbow')
    st.write(f'The estimated VaR at 95% confidence level for the next {time_horizon} days is:')
    st.text(var_95[0])

# Tab4 END===================================================================

# Tab5 START=================================================================

if st.session_state['homeTab'] == 'Updates':
    col1, col2 = st.columns(2)
    with col1:
        insHolders = yf.Ticker(ticker).institutional_holders
        st.subheader('Institutional Holders')
        st.caption('**All numbers in thousands**')
        st.dataframe(insHolders, use_container_width=True, hide_index = True)
    with col2:
        mutualHolders = yf.Ticker(ticker).mutualfund_holders
        st.subheader('MutualFund Holders')
        st.caption('**All numbers in thousands**')
        st.dataframe(mutualHolders, use_container_width=True, hide_index = True)
    filteredNews = yf.Ticker(ticker).news
    st.subheader('Latest Nexs', divider = 'rainbow')
    col3, col4 = st.columns(2)
    for news_item in filteredNews:
        try:
            with col3:
                st.subheader(news_item['title'])
                st.caption(news_item['publisher'])
                st.write(news_item['link'])
            with col4:
                st.image(news_item['thumbnail']['resolutions'][0]['url'], width = 100)
                st.markdown('---') 
        except Exception:
            st.text('')
# Tab5 END=================================================================